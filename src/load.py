import numpy as np
from PIL import Image
import numpy as np
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import json
import math
from scipy.io import loadmat
import os
import argparse
# This is the scene we are going to load.
# support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
### put your scene path ###
test_scene = "replica_v1/apartment_0/habitat/mesh_semantic.ply"
path = "replica_v1/apartment_0/habitat/info_semantic.json"
colors = loadmat('color101.mat')['colors']
colors = np.insert(colors, 0, values=np.array([[0,0,0]]), axis=0)


#### instance id to semantic id 
with open(path, "r") as f:
    annotations = json.load(f)

id_to_label = []
instance_id_to_semantic_label_id = np.array(annotations["id_to_label"])
for i in instance_id_to_semantic_label_id:
    if i < 0:
        id_to_label.append(0)
    else:
        id_to_label.append(i)
id_to_label = np.asarray(id_to_label)

######

sim_settings = {
    "scene": test_scene,  # Scene path
    "default_agent": 0,  # Index of the default agent
    "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
    "width": 512,  # Spatial resolution of the observations
    "height": 512,
    "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
}

# This function generates a config for the simulator.
# It contains two parts:
# one for the simulator backend
# one for the agent, where you can attach a bunch of sensors

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def transform_semantic(semantic_obs):
    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(colors.flatten())
    semantic_img.putdata((semantic_obs.flatten()).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = cv2.cvtColor(np.asarray(semantic_img), cv2.COLOR_RGB2BGR)
    return semantic_img



def make_simple_cfg(settings):


    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    rgb_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #depth snesor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    #semantic snesor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.orientation = [
        settings["sensor_pitch"],
        0.0,
        0.0,
    ]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec, semantic_sensor_spec]
    ##################################################################
    ### change the move_forward length or rotate angle
    ##################################################################
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.05) # 0.01 means 0.01 m
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=1.0) # 1.0 means 1 degree
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=1.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


cfg = make_simple_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)
   # obtain the default, discrete actions that an agent can perform
    # default action space contains 3 actions: move_forward, turn_left, and turn_right
action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
print("Discrete action space: ", action_names)


 # Todo
loaded_data = np.load('movingpath.npy')
position = []
for data in loaded_data:
    position.append((data[1], 0,data[0]))
 # initialize an agent
agent = sim.initialize_agent(sim_settings["default_agent"])
# # Set agent state
agent_state = habitat_sim.AgentState()
agent_state.position= np.array(position[0])
agent.set_state(agent_state)


def navigateAndSee(action="",data_root='data_collection/'):
    global count
    if action in action_names:
        observations = sim.step(action)
        #print("action: ", action)
        RGB_img = transform_rgb_bgr(observations["color_sensor"])
        SEIMEN_img =  transform_semantic(id_to_label[observations["semantic_sensor"]])
        index = np.where((SEIMEN_img[:,:,0]==b)*(SEIMEN_img[:,:,1]==g)*(SEIMEN_img[:,:,2]==r))
        # creat red frame
        height, width = 512,512 
        red_mask = np.zeros((height, width, 3), dtype=np.uint8)
        red_mask[:, :, 2] = 255  # 将红色通道设为255（其他通道为0）bgr
        #print("index",index)
        if len(index[0]) != 0:
            RGB_img[index] = cv2.addWeighted(RGB_img[index], 0.6, red_mask[index], 0.4, 50)
        cv2.imshow("RGB", RGB_img)
        videowriter.write(RGB_img)
        #cv2.imshow("RGB", transform_rgb_bgr(observations["color_sensor"]))
        agent_state = agent.get_state()
        sensor_state = agent_state.sensor_states['color_sensor']
        print("camera pose: x y z rw rx ry rz")
        print(sensor_state.position[0],sensor_state.position[1],sensor_state.position[2],  sensor_state.rotation.w, sensor_state.rotation.x, sensor_state.rotation.y, sensor_state.rotation.z)
        count += 1
    
        #cv2.imwrite(data_root + f"/{count}.png", RGB_img) 

def drive (position,i):

    if i == 0:
        v1 = np.array([0,0,-1]) 
    else:
        v1 = np.array([position[i][0]-position[i-1][0],0,position[i][2]-position[i-1][2]])
    v2 = np.array([position[i+1][0]-position[i][0],0,position[i+1][2]-position[i][2]])
    print("v1",v1)
    print("v2",v2)

    #rotate
    l1 = np.linalg.norm(v1)
    l2 = np.linalg.norm(v2)
    dot_product = np.dot(v1,v2)
    print("dot_product",dot_product)
    cos_theta = dot_product/(l1*l2)
    theta = np.arccos(cos_theta)
    print("theta",math.degrees(theta))
    cross = np.cross(v1,v2)
    print("cross",cross)
    if cross[1]>0:
        theta = -theta
    print("revise theta",math.degrees(theta))
    if theta>0:
        action = "turn_right"
        for i in range(int(math.degrees(theta))):
            navigateAndSee(action)
        #right
    else :
        action = "turn_left"
        theta = -theta
        for i in range(int(math.degrees(theta))):
            navigateAndSee(action)
        #left

    #move forward 
    action = "move_forward"
    length = np.linalg.norm(v2)
    forward_step = int(length/0.05) 
    for i in range(forward_step):
        navigateAndSee(action)
    #
    #x = sensor_state.position[0]
    #z = sensor_state.position[2]
    print(length)
    

target = {"refrigerator":(255, 0, 0),
        "rack":(0, 255, 133),
        "cushion":(255, 9, 92),
        "lamp":(160, 150, 20),
        "stair":(173,255,0),
        "cooktop":(7, 255, 224)} 



if __name__ == "__main__":
     # save video initial7, 255, 224)
    parser = argparse.ArgumentParser(description = 'Below are the params:')
    parser.add_argument('-p', type=str, default='rack',dest='target')
    args = parser.parse_args()
    target_rgb = target[args.target]
    print("target",target_rgb)
    r = target_rgb[0]
    g = target_rgb[1]
    b = target_rgb[2]
    # save video
    if not os.path.exists("video/"):
        os.makedirs("video/")
    video_path = "video/"  +args.target + ".mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(video_path, fourcc, 50, (512, 512))


    count = 0
    i=0
    #
    for i in range(len(position)-1):
        drive(position,i)
    videowriter.release()