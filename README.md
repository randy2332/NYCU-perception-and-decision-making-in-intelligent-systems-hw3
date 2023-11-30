# NYCU-perception-and-decision-making-in-intelligent-systems-hw3

On the established 2D map, the positions of the targets are marked. The targets are labeled with points in accordance with this sequence” rack , cushion, lamp, stair cooktop 

We also store the 2d map call “map.py”

```
python build2dmap.py
```

![Screenshot from 2023-11-30 00-36-35.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw3/blob/main/videoandpicture/map.png)

---

```
python main.py -p map.png -s 80
```

In this part you can choose the stepsize the 2d map you like to use.

Next, the system will ask you about your target. Afterwards, you will be presented with a 2D map to select your starting position.

We also output path point called “movingoath.npy”

Below is the result of RRT algorithm implementation

- cooktop

![cooktop.jpg](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw3/blob/main/videoandpicture/cooktop.jpg)

- cushion

![cushion.jpg](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw3/blob/main/videoandpicture/cushion.jpg)

- lamp

![lamp.jpg](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw3/blob/main/videoandpicture/lamp.jpg)

- rack

![rack.jpg](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw3/blob/main/videoandpicture/rack.jpg)

- stair

![stair.jpg](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw3/blob/main/videoandpicture/stair.jpg)

---

```
python load.py -p <target>
```

We highlight the target with a transparent mask while navigating, so that we can clearly visualize the target category

![686.png](https://github.com/randy2332/NYCU-perception-and-decision-making-in-intelligent-systems-hw3/blob/main/videoandpicture/686.png)
