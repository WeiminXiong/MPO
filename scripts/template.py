# ALFWorld specific template
ALFWORLD_TEMPLATE = """Please generate a step-by-step workflow for a house holding task:
<task>
{task}
</task>

The action list you can take: 
1. go to {{recep}}
2. task {{obj}} from {{recep}}
3. put {{obj}} in/on {{recep}}
4. open {{recep}}
5. close {{recep}}
6. toggle {{obj}} {{recep}}
7. clean {{obj}} with {{recep}}
8. heat {{obj}} with {{recep}}
9. cool {{obj}} with {{recep}}
where {{obj}} and {{recep}} correspond to objects and receptacles.

The generated workflow should be written in the following format:
<workflow>
Step 1: ...
Step 2: ...
...
</workflow>
"""

# SciWorld specific template
SCIWORLD_TEMPLATE = """Please generate a step-by-step workflow for a scientific task:
<task>
You are a helpful assistant to do some scientific experiment in an environment.
In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway.
{task}
</task>

You should explore the environment and find the items you need to complete the experiment.
You can teleport to any room in one step.
All containers in the environment have already been opened, you can directly get items from the containers.

The available actions are:
    open OBJ: open a container
    close OBJ: close a container
    activate OBJ: activate a device
    deactivate OBJ: deactivate a device
    connect OBJ to OBJ: connect electrical components
    disconnect OBJ: disconnect electrical components
    use OBJ [on OBJ]: use a device/item
    look around: describe the current room
    examine OBJ: describe an object in detail
    look at OBJ: describe a container's contents
    read OBJ: read a note or book
    move OBJ to OBJ: move an object to a container
    pick up OBJ: move an object to the inventory
    pour OBJ into OBJ: pour a liquid into a container
    mix OBJ: chemically mix a container
    teleport to LOC: teleport to a specific room
    focus on OBJ: signal intent on a task object
    wait: task no action for 10 steps
    wait1: task no action for a step

The generated workflow should be written in the following format:
<workflow>
Step 1: ...
Step 2: ...
...
</workflow>
"""