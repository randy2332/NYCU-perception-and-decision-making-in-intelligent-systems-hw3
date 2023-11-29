import numpy as np
import open3d as o3d
import argparse
import os
import copy
import time
import math
from sklearn.neighbors import NearestNeighbors
import cv2
import matplotlib.pyplot as plt

class Map:
    def __init__(self, point_path, color_path):
        self.point_path = point_path
        self.color_path = color_path
        self.points = np.load(self.point_path)
        self.colors = np.load(self.color_path)

    def get_pcd(self, pcd_path):
        pcd = o3d.io.read_point_cloud(pcd_path)
        #o3d.visualization.draw_geometries([pcd])
        return pcd

    def construct_pcd(self):
        pcd = o3d.geometry.PointCloud()
        self.points = self.points * 10000 / 255
        pcd.points = o3d.utility.Vector3dVector(self.points)
        pcd.colors = o3d.utility.Vector3dVector(self.colors)
        #o3d.visualization.draw_geometries([pcd])

        # filter the ceiling and the floor
        xyz_points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        ceiling_y = 0.0135
        floor_y = -1.35
        filtered_xyz_points = xyz_points[(xyz_points[:, 1] <= ceiling_y) 
                                        &(  floor_y <= xyz_points[:, 1])]
        filtered_colors = colors[(xyz_points[:, 1] <= ceiling_y) 
                                &(  floor_y <= xyz_points[:, 1])]
        pcd.points = o3d.utility.Vector3dVector(filtered_xyz_points)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        #o3d.visualization.draw_geometries([pcd])
        return pcd
    
    def construct_image(self, pcd):
        # create scatter plot
        pixel_per_inches = 1/plt.rcParams['figure.dpi'] #this calculates the pixel density based on the current Matplotlib figure's DPI (dots per inch).
        plt.figure(figsize=(1700 * pixel_per_inches, 
                            1100 * pixel_per_inches))
        points = np.asarray(pcd.points)
        plt.scatter(points[:, 2], points[:, 0],s=5, c = np.asarray(pcd.colors), marker='o')#Generates a scatter plot using the X and Z coordinates 

        # Set the scale
        plt.xlim(points[:, 2].min(), points[:, 2].max())
        plt.ylim(points[:, 0].min(), points[:, 0].max())
        plt.axis('off')
        plt.savefig('map.png', bbox_inches = 'tight', pad_inches = 0)

        plt.show()
        
        # reverse transform
        # x_restored = u * (points[:, 2].max() - points[:, 2].min()) + points[:, 2].min()
        # z_restored = (points[:, 0].max() - v) * (points[:, 0].max() - points[:, 0].min()) / (points[:, 0].max() - points[:, 0].min()) + points[:, 0].min()\

        

    def findproduct(self, pcd, product_names):
        product_colors = {
            'rack': np.array([0, 255, 133]),
            'cushion': np.array([255, 9, 92]),
            'lamp': np.array([160, 150, 20]),
            'stair': np.array([173, 255, 0]),
            'cooktop': np.array([7, 255, 224])
        }

        colors_np = np.asarray(pcd.colors)
        pixel_per_inches = 1/plt.rcParams['figure.dpi']
        plt.figure(figsize=(1700 * pixel_per_inches, 1100 * pixel_per_inches))
        points = np.asarray(pcd.points)

        plt.scatter(points[:, 2], points[:, 0], s=5, c=np.asarray(pcd.colors), marker='o')

        recorded_positions = []

        def on_double_click(event):
            nonlocal recorded_positions
            if event.dblclick:
                recorded_positions.append((event.xdata, event.ydata))
                plt.annotate(f"Pos {len(recorded_positions)}: {recorded_positions[-1]}", 
                             xy=(event.xdata, event.ydata),
                             xytext=(event.xdata + 10, event.ydata + 10),
                             textcoords="offset points",
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=0.5),
                             fontsize=8, color='black')

                plt.scatter(event.xdata, event.ydata, s=100, c='red', marker='o', alpha=0.5)
                plt.draw()
                
                if len(recorded_positions) == 5:
                    time.sleep(1)
                    plt.close()
        for product_name in product_names:
            product_color = product_colors[product_name]
            indices = np.where(np.all(colors_np*255 == product_color, axis=1))[0]

            if len(indices) > 0:
                plt.scatter(points[indices, 2], points[indices, 0], s=5, c='black', marker='o')
                plt.text(points[indices[0], 2], points[indices[0], 0], product_name, color='red', fontsize=16)


        plt.xlim(points[:, 2].min(), points[:, 2].max())
        plt.ylim(points[:, 0].min(), points[:, 0].max())
        plt.axis('off')
        fig = plt.gcf()
        fig.canvas.mpl_connect('button_press_event', on_double_click)
        plt.show()

        def mat_to_img(recorded_positions):
            img = cv2.imread("map.png",0)
            l,w = img.shape
            points = np.asarray(pcd.points)
            img_position = []
            mat_xlen = points[:, 2].max()- points[:, 2].min()
            mat_ylen = points[:, 0].max()- points[:, 0].min()
            print("mat_xlen:",mat_xlen)
            print("mat_ylen:",mat_ylen)
            print("matxmin:",points[:, 2].min())
            print("matymin:",points[:, 0].min())
            for position in recorded_positions:
                img_position_x = (position[0]-points[:, 2].min())/mat_xlen*w
                img_position_y = (1-((position[1]-points[:, 0].min())/mat_ylen))*l
                img_position.append((img_position_x, img_position_y))  

            return img_position 

        img_position = mat_to_img(recorded_positions)
        print("finish record")
        return img_position 

        

        

if __name__ == '__main__':
    point_path = "pointcloud/point.npy"
    color_path = "pointcloud/color01.npy"
    semantic_map = Map(point_path,color_path)
    pcd = semantic_map.construct_pcd()
    map_points = np.asarray(pcd.points)#this is 3d 
    map_colors = np.asarray(pcd.colors)#
    np.save("2D_semantic_map_points.npy",map_points)
    np.save("2D_semantic_map_colors.npy",map_colors)
    semantic_map.construct_image(pcd)
    product_names = ['rack', 'cushion', 'lamp', 'stair', 'cooktop']
    recordedimg_positions = semantic_map.findproduct(pcd,product_names)
    print("Recorded positions:", recordedimg_positions)
    positions_np = np.array(recordedimg_positions)
    #np.save('product_positions.npy', positions_np)