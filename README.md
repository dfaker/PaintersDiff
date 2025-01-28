# Painter's Diff
![image](https://github.com/user-attachments/assets/a1c24893-b2d2-4fe0-8e0a-52cf1281e2dd)

Iterative guided painting and drawing by webcam diff feedback.

Registers a webcam, projector and canvas to guide the progress of a painting or drawing reaching a match to a target image.

Upon loading the program expects a projector and webcam to be pointed at the canvas.

Initially you will be asked to select a target image.
you will be asked to select the canvas corners on the projector displau.
You will then be asked to selected the same corners on the webcam display.

The program will register these sets of points, and being projecting the initial diff image, with sections in need of darkening appeating as red, and lightening as blue.

Make a mark on the canvas in wherever the darkest area is indicated (initially the brightest red tone) and press recauclate to use that as the reference back on the canvas.

Follow the projected guidance, and when you think you've made progress move your hands, pends and brushes away from the canvas and click 'Recaclulate' to get an updated guidance image.  

Tweak the excessive number of values and sliders to see results - most are experiments I've not found completely useless.

Even medium lighting on the canvas is essential.

 
