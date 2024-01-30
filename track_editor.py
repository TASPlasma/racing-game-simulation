import tkinter as tk
from curve import Curve
from math_types import Vector2f
from simulate import Trajectory
from tkinter import Canvas, Radiobutton

class TrackEditor:
    def __init__(self, root):
        self.inner_curve = Curve('blue', [])
        self.outer_curve = Curve('red', [])

        self.root = root
        self.root.title("Track Editor")

        self.canvas = Canvas(self.root, width=400, height=400, bg="white")
        self.canvas.grid(column=0, row=0)

        self.radio_var = tk.IntVar()
        self.radio_var.set(1)

        self.inner_radio_button = Radiobutton(
            self.root,
            text="Inner Curve",
            variable=self.radio_var,
            value=1
        )
        self.inner_radio_button.grid(column=0, row=1)

        self.outer_radio_button = Radiobutton(
            self.root,
            text="Outer Curve",
            variable=self.radio_var,
            value=2
        )
        self.outer_radio_button.grid(column=0, row=2)

        self.click_only_mode_var = tk.BooleanVar()
        self.click_only_mode_var.set(False)
        self.click_only_mode_checkbox = tk.Checkbutton(
            self.root,
            text="Click Only Mode",
            variable=self.click_only_mode_var
        )
        self.click_only_mode_checkbox.grid(column=0, row=3)

        self.drift_dir_var = tk.IntVar()
        self.drift_dir_var.set(0)
        self.drift_left_btn = Radiobutton(
            self.root,
            text="Drift Left",
            variable=self.drift_dir_var,
            value=-1,
            command=self.update_stick_dir
        )
        self.drift_left_btn.grid(column=1, row=1)

        self.no_drift_btn = Radiobutton(
            self.root,
            text="No Drift",
            variable=self.drift_dir_var,
            value=0,
            command=self.update_stick_dir
        )
        self.no_drift_btn.grid(column=2, row=1)

        self.drift_right_btn = Radiobutton(
            self.root,
            text="Drift Right",
            variable=self.drift_dir_var,
            value=1,
            command=self.update_stick_dir
        )
        self.drift_right_btn.grid(column=3, row=1)

        self.pos_inputs_var = tk.IntVar(value=1) # if 1, inputs are positive, else negative
        self.pos_inputs_btn = tk.Checkbutton(
            self.root,
            text="Positive Inputs",
            variable=self.pos_inputs_var,
            onvalue=1,
            offvalue=-1, 
        )
        self.pos_inputs_btn.grid(column=2, row=2)

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        self.root.bind("<KeyPress>", self.update_pos)
        # self.root.bind("<KeyRelease>", self.update_pos)

        self.save_track_btn = tk.Button(
            self.root,
            text="Save Track",
            command=self.save_track
        )
        self.save_track_btn.grid(column=0, row=4)

        self.load_track_btn = tk.Button(
            self.root,
            text="Load Track",
            command=self.load_track
        )
        self.load_track_btn.grid(column=0, row=5)

        self.clear_track_btn = tk.Button(
            self.root,
            text="Clear Track",
            command=self.clear_track
        )
        self.clear_track_btn.grid(column=0, row=6)

        self.drawing = False
        self.first_click = False

        self.char_pos = Trajectory(dir = 'neutral')
        self.draw_char_point(self.char_pos.pos)

    def update_stick_dir(self):
        drift_dir = self.drift_dir_var.get()
        if drift_dir == -1:
            self.char_pos.dir = 'left'
        elif drift_dir == 0:
            self.char_pos.dir = 'neutral'
        else:
            self.char_pos.dir = 'right'

    def clear_track(self):
        self.canvas.delete("all")
        self.inner_curve.clear()
        self.outer_curve.clear()
        self.draw_char_point(self.char_pos.pos)

    def polygon_from_pts(self, points):
        # points: a list of (x, y) which are vertices of a polygon
        pass

    def chosen_curve(self) -> Curve:
        if self.radio_var.get() == 1:
            return self.inner_curve
        return self.outer_curve

    def start_drawing(self, event):
        self.drawing = True
        pos = Vector2f(event.x, event.y)
        self.draw_point(pos, self.chosen_curve())

    def stop_drawing(self, event):
        self.drawing = False
        self.first_click = False

    def draw_char_point(self, pos):
        x, y = pos.x, pos.y
        self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill = "purple")

    def update_pos(self, event):
        print(event)
        if event.char == 'a':
            # enter left drift mode
            self.drift_dir_var.set(-1)
            self.char_pos.dir = 'left'

        elif event.char == 's':
            # enter no drift mode
            self.drift_dir_var.set(0)
            self.char_pos.dir = 'neutral'

        elif event.char == 'd':
            # enter right drift mode
            self.drift_dir_var.set(1)
            self.char_pos.dir = 'right'

        elif event.char == 'f':
            # toggle input sign value
            self.pos_inputs_var.set(-1) if self.pos_inputs_var.get() == 1 else self.pos_inputs_var.set(1)

        elif event.char.isdigit():
            input = self.pos_inputs_var.get() * int(event.char)
            new_vel = self.char_pos.update_vel(input)
            print(f"Input: {input}, new velocity: {new_vel}, stick direction: {self.char_pos.dir}")
            self.char_pos.vel = new_vel
            self.char_pos.pos = self.char_pos.update_pos(input)
            self.draw_char_point(self.char_pos.pos)


    def draw_point(self, pos, curve: Curve):
        if self.drawing and not self.first_click:
            x, y = pos.x, pos.y

            # If click_only_mode is enabled, append only the first point clicked
            if self.click_only_mode_var.get() and not self.first_click:
                self.first_click = True

            # If click_only_mode is not enabled or it's the first point, draw it
            if not self.click_only_mode_var.get() or self.first_click:
                # Add the point to the curve
                curve.points += [(x, y)]

                # Draw a point on the canvas
                self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=curve.color)

    def save_track(self):
        # Save the inner and outer curves to a file
        with open("track.txt", "w") as file:
            file.write("Inner Curve:\n")
            for point in self.inner_curve:
                file.write(f"{point[0]}, {point[1]}\n")

            file.write("Outer Curve:\n")
            for point in self.outer_curve:
                file.write(f"{point[0]}, {point[1]}\n")

    def load_track(self):
        file_path = "track.txt"
        if file_path:
            self.inner_curve.clear()
            self.outer_curve.clear()
            with open(file_path, "r") as file:
                current_curve = None
                for line in file:
                    line = line.strip()
                    if line == "Inner Curve:":
                        current_curve = self.inner_curve
                    elif line == "Outer Curve:":
                        current_curve = self.outer_curve
                    else:
                        x, y = map(float, line.split(","))
                        current_curve.points += [(x, y)]
            # Redraw the loaded track on the canvas
            self.redraw_loaded_track()
            # self.canvas.create_polygon(self.inner_curve)

    def redraw_loaded_track(self):
        # Clear the canvas and redraw all points
        self.canvas.delete("all")
        self.draw_char_point(self.char_pos.pos)
        for point in self.inner_curve:
            self.canvas.create_oval(
                point[0] - 2,
                point[1] - 2, 
                point[0] + 2, 
                point[1] + 2, 
                fill=self.inner_curve.color
                )
        for point in self.outer_curve:
            self.canvas.create_oval(
                point[0] - 2, 
                point[1] - 2, 
                point[0] + 2, 
                point[1] + 2, 
                fill=self.outer_curve.color
                )

if __name__ == "__main__":
    root = tk.Tk()
    app = TrackEditor(root)
    # root.bind("<Motion>", app.draw_point())
    root.mainloop()
