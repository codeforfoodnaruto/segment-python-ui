import os
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

import cv2
import torch
import numpy as np

from segment_anything import sam_model_registry, SamPredictor

class SamSegGUI:
    def __init__(self, root, video_path, sam_checkpoint, model_type="vit_h"):
        self.root = root
        self.root.title("Basketball Segmentation with SAM (Tkinter)")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print("Cannot open video!")
            raise ValueError("Invalid video path.")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.predictor = SamPredictor(sam)

        self.save_dir = "./dataset"
        os.makedirs(self.save_dir, exist_ok=True)

        self.frame_index = 0
        self.skip_frames = 20

        self.frame_bgr = None
        self.frame_rgb = None
        self.read_new_frame()

        self.clicked_points = []
        self.clicked_labels = []

        if self.frame_rgb is not None:
            self.img_w = self.frame_rgb.shape[1]
            self.img_h = self.frame_rgb.shape[0]
        else:
            self.img_w, self.img_h = (640, 480)

        self.canvas = tk.Canvas(root, width=self.img_w, height=self.img_h)
        self.canvas.pack(side=tk.TOP, padx=5, pady=5)

        btn_frame = tk.Frame(root)
        btn_frame.pack(side=tk.TOP, pady=5)

        self.segment_btn = ttk.Button(btn_frame, text="Segment", command=self.run_segmentation)
        self.segment_btn.pack(side=tk.LEFT, padx=5)

        self.next_btn = ttk.Button(btn_frame, text="Next Frame", command=self.on_next_frame)
        self.next_btn.pack(side=tk.LEFT, padx=5)

        root.bind("<space>", lambda event: self.on_next_frame())
        root.bind("e", lambda event: self.run_segmentation())
        root.bind("q", lambda event: self.redo())

        self.tk_image = None
        if self.frame_rgb is not None:
            self.update_tk_image(self.frame_rgb)

        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

    def read_new_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print("No more frames or failed to read.")
            self.frame_bgr = None
            self.frame_rgb = None
            return

        self.frame_bgr = frame
        self.frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(self.frame_rgb)

    def on_next_frame(self):
        self.clicked_points.clear()
        self.clicked_labels.clear()

        frames_to_skip = self.skip_frames - 1
        for _ in range(frames_to_skip):
            ret = self.cap.grab()
            if not ret:
                print("Video ended during skip.")
                self.frame_bgr = None
                self.frame_rgb = None
                break

        if self.frame_bgr is not None:
            self.read_new_frame()
            if self.frame_rgb is None:
                self.disable_buttons()
                return
            self.frame_index += self.skip_frames
            self.update_tk_image(self.frame_rgb)
        else:
            self.disable_buttons()

    def disable_buttons(self):
        self.next_btn.config(state=tk.DISABLED)
        self.segment_btn.config(state=tk.DISABLED)

    def update_tk_image(self, img_rgb):
        self.canvas.delete("all")
        from PIL import Image, ImageTk
        pil_image = Image.fromarray(img_rgb)
        self.tk_image = ImageTk.PhotoImage(image=pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def on_left_click(self, event):
        if self.frame_rgb is None:
            return
        x, y = event.x, event.y
        self.clicked_points.append([x, y])
        self.clicked_labels.append(1)
        self.draw_click_marker(x, y, color="green")

    def on_right_click(self, event):
        if self.frame_rgb is None:
            return
        x, y = event.x, event.y
        self.clicked_points.append([x, y])
        self.clicked_labels.append(0)
        self.draw_click_marker(x, y, color="red")

    def draw_click_marker(self, x, y, color="green"):
        r = 5
        self.canvas.create_oval(x-r, y-r, x+r, y+r, outline=color, width=2)

    def run_segmentation(self):
        if self.frame_rgb is None:
            return
        if len(self.clicked_points) == 0:
            print("No points provided!")
            return

        point_coords = np.array(self.clicked_points)
        point_labels = np.array(self.clicked_labels)

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True
        )
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]

        overlay_rgb = self.apply_mask_to_image(self.frame_rgb, best_mask,
                                               color=(0, 255, 0), alpha=0.5)
        self.update_tk_image(overlay_rgb)

        self.save_segmented_image(overlay_rgb)

    def apply_mask_to_image(self, img_rgb, mask, color=(0,255,0), alpha=0.5):
        overlay = img_rgb.copy()
        c_r, c_g, c_b = color
        overlay[mask > 0] = (
            overlay[mask > 0] * (1 - alpha) + np.array([c_r, c_g, c_b]) * alpha
        ).astype(np.uint8)
        return overlay

    def save_segmented_image(self, overlay_rgb):
        filename = f"frame_{self.frame_index:04d}.png"
        save_path = os.path.join(self.save_dir, filename)
        overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, overlay_bgr)
        print(f"Saved segmented result to: {save_path}")

    def redo(self):
        if self.frame_rgb is None:
            return
        self.clicked_points.clear()
        self.clicked_labels.clear()
        self.update_tk_image(self.frame_rgb)

def main():
    root = tk.Tk()

    video_path = "basketball.mp4"
    sam_checkpoint = "sam_vit_h_4b8939.pth"

    app = SamSegGUI(root, video_path, sam_checkpoint, model_type="vit_h")

    root.mainloop()

if __name__ == "__main__":
    main()
