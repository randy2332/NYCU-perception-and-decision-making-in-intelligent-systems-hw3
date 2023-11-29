import cv2
import numpy as np
import math
import random
import argparse
import os
import open3d as o3d
class Nodes:
    """Class to store the RRT graph"""
    def __init__(self, x,y):
        self.x = x
        self.y = y
        self.parent_x = []
        self.parent_y = []

# check collision
def collision(x1,y1,x2,y2):
    color=[]
    x = list(np.arange(x1,x2,(x2-x1)/100))
    y = list(((y2-y1)/(x2-x1))*(x-x1) + y1)
    #print("collision",x,y)
    for i in range(len(x)):
        #print(int(x[i]),int(y[i]))
        color.append(img[int(y[i]),int(x[i])])
        #print("color:",img[int(y[i]),int(x[i])])
        #print(color)
    if all(x == 255 for x in color):
        return False  #colli
    else:
        return True

# check the  collision with obstacle and trim
def check_collision(x1,y1,x2,y2):
    _,theta = dist_and_angle(x2,y2,x1,y1)
    x=x2 + stepSize*np.cos(theta)
    y=y2 + stepSize*np.sin(theta)
    #print(x2,y2,x1,y1)
    #print("theta",theta)
    #print("check_collision",x,y)

    # trim the branch if its going out of image area
    # print("Image shape",img.shape)
    hy,hx=img.shape
    if y<0 or y>hy or x<0 or x>hx:
        print("Point out of image bound")
        directCon = False
        nodeCon = False
    else:
        # check direct connection
        if collision(x,y,end[0],end[1]):
            directCon = False
        else:
            directCon=True

        # check connection between two nodes
        if collision(x,y,x2,y2):
            nodeCon = False
        else:
            nodeCon = True

    return(x,y,directCon,nodeCon)

# return dist and angle b/w new point and nearest node
def dist_and_angle(x1,y1,x2,y2):
    dist = math.sqrt( ((x1-x2)**2)+((y1-y2)**2) )
    angle = math.atan2(y2-y1, x2-x1)
    return(dist,angle)

# return the neaerst node index
def nearest_node(x,y):
    temp_dist=[]
    for i in range(len(node_list)):
        dist,_ = dist_and_angle(x,y,node_list[i].x,node_list[i].y)
        temp_dist.append(dist)
    return temp_dist.index(min(temp_dist))

# generate a random point in the image space
def rnd_point(h,l):
    new_y = random.randint(0, h)
    new_x = random.randint(0, l)
    return (new_x,new_y)


def RRT(img, img2, start, end, stepSize):
    h,l= img.shape # dim of the loaded image
    # print(img.shape) # (384, 683)
    # print(h,l)

    # insert the starting point in the node class
    # node_list = [0] # list to store all the node points         
    node_list[0] = Nodes(start[0],start[1])
    node_list[0].parent_x.append(start[0])
    node_list[0].parent_y.append(start[1])

    # display start and end
    cv2.circle(img2, (start[0],start[1]), 5,(0,0,255),thickness=3, lineType=8)
    cv2.circle(img2, (int(end[0]),int(end[1])), 5,(0,0,255),thickness=3, lineType=8)

    i=1
    pathFound = False
    while pathFound==False:
        nx,ny = rnd_point(h,l)
        print("Random points:",nx,ny)

        nearest_ind = nearest_node(nx,ny)
        nearest_x = node_list[nearest_ind].x
        nearest_y = node_list[nearest_ind].y
        print("Nearest node coordinates:",nearest_x,nearest_y)

        #check direct connection
        tx,ty,directCon,nodeCon = check_collision(nx,ny,nearest_x,nearest_y)
        print("Check collision:",tx,ty,directCon,nodeCon)

        if directCon and nodeCon:
            print("Node can connect directly with end")
            node_list.append(i)
            node_list[i] = Nodes(tx,ty)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)

            cv2.circle(img2, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
            cv2.line(img2, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
            cv2.line(img2, (int(tx),int(ty)), (int(end[0]),int(end[1])), (255,0,0), thickness=2, lineType=8)

            #store the moving path(x,y) 
            movingpath = []
            print("Path has been found")
            movingpath.append((start[0],start[1]))
            #print("parent_x",node_list[i].parent_x)
            for j in range(1,len(node_list[i].parent_x)):
                movingpath.append((node_list[i].parent_x[j],node_list[i].parent_y[j]))
                cv2.line(img2, (int(node_list[i].parent_x[j-1]),int(node_list[i].parent_y[j-1])), (int(node_list[i].parent_x[j]),int(node_list[i].parent_y[j])), (255,0,0), thickness=2, lineType=8)
            movingpath.append((end[0],end[1]))
            # cv2.waitKey(1)
            
            cv2.imwrite("media/"+str(i)+".jpg",img2)
            cv2.imwrite("out.jpg",img2)
            print(movingpath)
            img_to_mat(movingpath)

            break

        elif nodeCon:
            print("Nodes connected")
            node_list.append(i)
            node_list[i] = Nodes(tx,ty)
            node_list[i].parent_x = node_list[nearest_ind].parent_x.copy()
            node_list[i].parent_y = node_list[nearest_ind].parent_y.copy()
            # print(i)
            # print(node_list[nearest_ind].parent_y)
            node_list[i].parent_x.append(tx)
            node_list[i].parent_y.append(ty)
            i=i+1
            # display
            cv2.circle(img2, (int(tx),int(ty)), 2,(0,0,255),thickness=3, lineType=8)
            cv2.line(img2, (int(tx),int(ty)), (int(node_list[nearest_ind].x),int(node_list[nearest_ind].y)), (0,255,0), thickness=1, lineType=8)
            cv2.imwrite("media/"+str(i)+".jpg",img2)
            cv2.imshow("sdc",img2)
            cv2.waitKey(1)
            continue

        else:
            print("No direct con. and no node con. :( Generating new rnd numbers")
            continue

def draw_circle(event,x,y,flags,param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img2,(x,y),5,(255,0,0),-1)
        coordinates.append(x)
        coordinates.append(y)
def img_to_mat(movingpath):
    img = cv2.imread("map.png",0)
    l,w = img.shape

    mat_xlen = 14.861425160934004
    mat_ylen = 9.31226060920138
    matxmin = -4.948978826738077
    matymin = -3.0890559048247224
    mat_position = []
    for position in movingpath:
        position_x = position[0]/w*mat_xlen+matxmin
        position_y = (1-(position[1]/l))*mat_ylen+matymin
        mat_position.append((position_x,position_y))
    print("mat_position",mat_position)
    mat_position = np.array(mat_position)
    np.save('movingpath.npy',mat_position )

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Below are the params:')
    parser.add_argument('-p', type=str, default='map.png',metavar='ImagePath', action='store', dest='imagePath',
                    help='Path of the image containing mazes')
    parser.add_argument('-s', type=int, default=10,metavar='Stepsize', action='store', dest='stepSize',
                    help='Step-size to be used for RRT branches')
   
    args = parser.parse_args()


    '''read .npy that we have save to define position in img'''
    loaded_data = np.load('product_positions.npy')
    rack = loaded_data[0]
    cushion = loaded_data[1]
    lamp = loaded_data[2]
    stair = loaded_data[3]
    cooktop = loaded_data[4]
    target= input("Enter the target that you like to find:")
    if target == "rack":
        end = rack
    elif target == "cushion":
        end = cushion
    elif target =="lamp":
        end = lamp
    elif target == "stair":
        end = stair
    elif target == "cooktop":
        end = cooktop
    else :
        print("can't search")

    '''read img'''
    img = cv2.imread(args.imagePath,0) # load grayscale maze image
    img2 = cv2.imread(args.imagePath) # load colored maze image
    stepSize = args.stepSize # stepsize for RRT
    node_list = [0] # list to store all the node points

    coordinates=[]
    print("Select start  points by double clicking, press 'escape' to exit")
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)
    while(1):
        cv2.imshow('image',img2)
        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break
    start=(coordinates[0],coordinates[1])
    #end=(coordinates[2],coordinates[3])


    RRT(img, img2, start, end, stepSize)
