# NYCU-perception-and-decision-making-in-intelligent-systems-hw3

On the established 2D map, the positions of the targets are marked. The targets are labeled with points in accordance with this sequence” rack , cushion, lamp, stair cooktop 

We also store the 2d map call “map.py”

```
python build2dmap.py
```

![Screenshot from 2023-11-30 00-36-35.png](NYCU-perception-and-decision-making-in-intelligent%20248819f27a224478b8eafd403203f4e1/Screenshot_from_2023-11-30_00-36-35.png)

---

```
python main.py -p map.png -s 80
```

In this part you can choose the stepsize the 2d map you like to use.

Next, the system will ask you about your target. Afterwards, you will be presented with a 2D map to select your starting position.

We also output path point called “movingoath.npy”

Below is the result of RRT algorithm

- cooktop

![cooktop.jpg](NYCU-perception-and-decision-making-in-intelligent%20248819f27a224478b8eafd403203f4e1/cooktop.jpg)

- cushion

![cushion.jpg](NYCU-perception-and-decision-making-in-intelligent%20248819f27a224478b8eafd403203f4e1/cushion.jpg)

- lamp

![lamp.jpg](NYCU-perception-and-decision-making-in-intelligent%20248819f27a224478b8eafd403203f4e1/lamp.jpg)

- rack

![rack.jpg](NYCU-perception-and-decision-making-in-intelligent%20248819f27a224478b8eafd403203f4e1/rack.jpg)

- stair

![stair.jpg](NYCU-perception-and-decision-making-in-intelligent%20248819f27a224478b8eafd403203f4e1/stair.jpg)

---

```
python load.py -p <target>
```

We highlight the target with a transparent mask while navigating, so that we can clearly visualize the target category

![686.png](NYCU-perception-and-decision-making-in-intelligent%20248819f27a224478b8eafd403203f4e1/686.png)