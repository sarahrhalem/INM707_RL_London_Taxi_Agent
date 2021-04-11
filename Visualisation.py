# -*- coding: utf-8 -*-
"""
@author: Sarah Rhalem

Module to visulaize City and trajectory taken by agent
"""
import matplotlib
import matplotlib.pyplot as plt
import City

def plot_vis(inputs,fig_size=10):
    circles = inputs['circles']
    roads   = inputs['roads']
    current = inputs['current_loc']
    target  = inputs['target_loc']
    start   = inputs['start_loc']
    n_zones = inputs['n_zones']
    radius = inputs['max_radius']
    oneway_roads = inputs['oneway_roads']
    trajectory =[]
    traj_zone =[]
    if 'trajectory_road' in inputs:
        trajectory =inputs['trajectory_road']
    if 'trajectory_zone' in inputs:
        traj_zone =inputs['trajectory_zone']
        
    fig, ax = plt.subplots(figsize=(fig_size,fig_size) )
    ax.set_xlim((-fig_size, fig_size))
    ax.set_ylim((-fig_size, fig_size))
    plt.title("London Taxi Environment", fontsize=18)
    ax.axis("equal")
    ax.axis("off")
    
    for circle in circles:
        ax.add_patch(circle)
    for road in roads:
        ax.add_collection(road)
    
    arc =matplotlib.patches.Arc((0,0),0,0,color='r')
    for item in traj_zone:
        arc =matplotlib.patches.Arc((0,0),item[0],item[0],
                                            angle=item[1],theta1=0, theta2=item[2],color='r')
        ax.add_patch(arc)
    
    currentp=plt.scatter(current[0],current[1],color='orange',s=100, marker="v", label="Driver")
    #start=plt.scatter(start[0],start[1],color='black',s=100, marker="H", label="Start")
    targetp=plt.scatter(target[0],target[1],color='green',s=100, marker="D", label="Target")
    
    

    for item in oneway_roads:
        
        #ax.annotate("", xy=item[0], xytext=item[1],
                                  #arrowprops={'arrowstyle': '-|>', 'lw': 6, 'ec': 'g'})
        ow_r = plt.arrow(item[0][0],item[0][1],item[1][0]-item[0][0],item[1][1]-item[0][1],width=0.125,
                         length_includes_head=True,color='g')
    path = plt.arrow(current[0],current[1],0,0,width=0.0001,length_includes_head=True,color='r')
    for item in trajectory:
        path = plt.arrow(item[1][0],item[1][1],item[0][0]-item[1][0],item[0][1]-item[1][1],width=0.075,
                         length_includes_head=True,color='r')

    ax.legend([circle, road,currentp, targetp,ow_r,path,arc],
              ["zones","roads", "driver","target","one way road","trajectory"], 
              markerscale=0.5, bbox_to_anchor=(0.7,0.7))
    
    
    
    plt.show();
    
    return 