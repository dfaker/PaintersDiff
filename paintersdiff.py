import cv2
import numpy as np
import cv2.ximgproc
import time
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mb

import screeninfo
import traceback
import os
import datetime
import tkinter.filedialog as filedialog

import threading

import scipy.interpolate as interpolate
from skimage.metrics import structural_similarity as ssim

from tkinter import colorchooser
import json

from scipy.ndimage import gaussian_filter
from collections import deque

import math
from colorsys import hsv_to_rgb



"""
im = cv2.imread('..\\KSB4PC26VEJE1Q3DYPNYC28X00.jpeg')


im = cv2.cvtColor(im,cv2.COLOR_RGB2HLS)
im = cv2.cvtColor(im,cv2.COLOR_HLS2RGB)

cv2.imshow('im',im)
cv2.waitKey(0)

exit()
"""
cv2.ocl.setUseOpenCL(True)


def canny_distance_pyramid(img1, img2):
    img1 = img1.astype(np.float32) / 255.0
    img2 = img2.astype(np.float32) / 255.0

    # Compute signed difference in lightness
    diff = img2 - img1  # Range: -1 to 1

    # Edge detection improvements
    # Calculate gradient magnitude instead of binary edges
    sobel_x = cv2.Sobel(img2, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img2, cv2.CV_32F, 0, 1, ksize=3)
    edge_strength = np.sqrt(sobel_x**2 + sobel_y**2)
    edge_weights = np.clip(edge_strength / edge_strength.max(), 0, 1)

    # Calculate canvas edges to avoid reinforcing existing features
    canvas_edges = cv2.Sobel(img1, cv2.CV_32F, 1, 1, ksize=3)
    edge_weights *= np.clip(1 - np.abs(canvas_edges), 0.2, 1)

    # Darkness priority mask with non-linear scaling
    darkness_mask = 1 - np.power(img2, 0.5)  # Prioritize darker regions

    # Combined weight map with dynamic balancing
    edge_ratio = 0.4  # Reduced from 0.5 to decrease edge dominance
    weight_map = (edge_weights * edge_ratio + 
                 darkness_mask * (1 - edge_ratio))

    # Apply weight map to difference
    weighted_diff = diff * weight_map

    # Multi-scale processing with edge preservation
    pyramid_levels = 3
    pyramid = [weighted_diff]
    for _ in range(pyramid_levels - 1):
        # Use bilateral filtering for better edge preservation
        smoothed = cv2.bilateralFilter(pyramid[-1], 9, 75, 75)
        pyramid.append(cv2.pyrDown(smoothed))

    # Upsample and combine with edge-aware blending
    for i in range(len(pyramid) - 1, 0, -1):
        expanded = cv2.pyrUp(pyramid[i])
        h, w = pyramid[i - 1].shape
        # Use edge weights to control blending
        blend_mask = cv2.resize(edge_weights, (w, h))
        pyramid[i - 1] = pyramid[i - 1] * (1 - blend_mask) + expanded[:h, :w] * blend_mask

    # Final weighted difference
    final_diff = pyramid[0]

    # Adaptive thresholding to eliminate small fluctuations
    abs_diff = np.abs(final_diff)
    adaptive_threshold = 0.1 + 0.2 * (1 - darkness_mask)
    final_diff = np.where(abs_diff > adaptive_threshold, final_diff, 0)

    # Scale and invert
    final_diff = np.clip(final_diff * 255, -255, 255)
    final_diff = -final_diff  # Invert around zero

    return final_diff

def signed_loss(img1, img2):
    return img1-img2

def signed_luminance_diff(img1, img2):
    # Positive means img1 is brighter, negative means img1 is darker
    # Using standard luminance weights
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR )
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR )

    weights = np.array([0.2989, 0.5870, 0.1140])
    lum1 = np.dot(img1, weights)
    lum2 = np.dot(img2, weights)
    return lum1 - lum2

def signed_intensity_diff(img1, img2):
    # Simpler version using average across channels
    if img1.ndim == 2:
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR )
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR )
    return (img1.mean(axis=2) - img2.mean(axis=2)).astype(int)

def signed_structural_loss(img1, img2, k1=0.01, k2=0.03, L=255, window_size=11):
   # Returns signed differences based on local structure comparison
   from scipy.ndimage import uniform_filter

   if img1.ndim == 2:
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR )
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR )

   # Convert to float
   img1 = img1.astype(float)
   img2 = img2.astype(float)
   
   # Compute means and variances using local windows
   mu1 = uniform_filter(img1, size=window_size)
   mu2 = uniform_filter(img2, size=window_size)
   
   sigma1 = uniform_filter(img1 ** 2, size=window_size) - mu1 ** 2
   sigma2 = uniform_filter(img2 ** 2, size=window_size) - mu2 ** 2
   sigma12 = uniform_filter(img1 * img2, size=window_size) - mu1 * mu2
   
   c1 = (k1 * L) ** 2
   c2 = (k2 * L) ** 2
   
   # Structure comparison that preserves sign
   diff = (2 * sigma12 + c2) / (sigma1 + sigma2 + c2)
   diff *= np.sign(mu1 - mu2)
   
   return np.clip(diff * 255, -255, 255)


def gradient_aware_loss(img1, img2, sigma=1.0):
   # Compares local gradients for edge-aware differences

   if img1.ndim == 2:
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR )
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR )
       
   gx1 = gaussian_filter(img1, sigma, order=[1,0,0])
   gy1 = gaussian_filter(img1, sigma, order=[0,1,0])
   gx2 = gaussian_filter(img2, sigma, order=[1,0,0])
   gy2 = gaussian_filter(img2, sigma, order=[0,1,0])
   
   grad_diff = np.sqrt((gx1-gx2)**2 + (gy1-gy2)**2)
   return np.clip(grad_diff * np.sign(img1.astype(float) - img2.astype(float)), -255, 255)

def frequency_loss(img1, img2):
   # Compares frequency domain differences
   from scipy.fft import fft2, ifft2
   
   f1 = fft2(img1.astype(float))
   f2 = fft2(img2.astype(float))
   diff = np.real(ifft2(f1 - f2))
   return np.clip(diff, -255, 255)

def multiscale_loss(img1, img2, levels=4):
   # Combines differences at multiple scales
   
   diff = np.zeros_like(img1, dtype=float)
   for i in range(levels):
       sigma = 2**i
       img1_blur = gaussian_filter(img1.astype(float), sigma)
       img2_blur = gaussian_filter(img2.astype(float), sigma)
       diff += (img1_blur - img2_blur) / levels
       
   return np.clip(diff, -255, 255)


"""
img1 = cv2.imread('original.jpg')
img2 = cv2.imread('current.jpg')

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY )
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY )

out = signed_structural_loss(img1,img2)
cv2.imshow('out',out.astype(np.uint8))
cv2.waitKey(0)
exit()
"""

losses = {
    "Signed loss":signed_loss,
    "Signed luminance diff":signed_luminance_diff,
    "Signed intensity diff":signed_intensity_diff,
    "Frequency domain loss":frequency_loss,
    "Signed structural loss":signed_structural_loss,
    "Gradient aware loss":gradient_aware_loss,
    "Multiscale loss l2":lambda a,b:multiscale_loss(a,b,levels=2),
    "Multiscale loss l3":lambda a,b:multiscale_loss(a,b,levels=3),
    "Multiscale loss l4":lambda a,b:multiscale_loss(a,b,levels=4),
    "Multiscale loss l5":lambda a,b:multiscale_loss(a,b,levels=5),
    "Multiscale canny distance pyramid":canny_distance_pyramid,
}

def create_blackwhite_lut(img_a, img_b, bit_depth=8):

    min_a, max_a = np.min(img_a), np.max(img_a)
    min_b, max_b = np.min(img_b), np.max(img_b)
    # Create LUT
    size = 2**bit_depth
    lut = np.arange(size, dtype=np.float32)
    # Apply black/white point mapping
    scale = (max_b - min_b) / (max_a - min_a)

    lut = np.clip(((lut - min_a) * scale + min_b), 0, size-1)

    return lut

def RGB2GreyCustom(img):
    return 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]

def posterize(image, levels):

    if levels<3:
        levels=3
    if levels>=256:
        levels=256

    # Calculate the scaling factor
    factor = 255 / (levels - 1)
    
    # Quantize to fewer levels
    temp = np.round(image / factor) * factor
    
    # Ensure output is in valid range
    return np.clip(temp, 0, 255).astype(np.uint8)

class HueRangeSelector(tk.Canvas):
    def __init__(self, master, radius=150, **kwargs):
        super().__init__(master, width=radius*2 + 20, height=radius*2 + 20, **kwargs)
        self.radius = radius
        self.center = (radius + 10, radius + 10)
        
        # Selection range in degrees (0-360)
        self.start_angle = 0
        self.end_angle = 360
        self.range_size = 180  # Store the range size for midpoint dragging
        self.dragging = None
        
        # Draw the initial wheel
        self.draw_wheel()
        self.draw_selection()
        
        # Bind mouse events
        self.bind('<Button-1>', self.on_click)
        self.bind('<B1-Motion>', self.on_drag)
        self.bind('<ButtonRelease-1>', self.on_release)
        self.bind('<Button-3>', self.on_right_click)  # Right click for range adjustment
        self.bind('<B3-Motion>', self.on_range_adjust)
        
        # Callback for when selection changes
        self.selection_callback = None

    def draw_wheel(self):
        """Draw the hue wheel with segments."""
        segments = 360
        for i in range(segments):
            angle1 = math.radians(i)
            angle2 = math.radians((i + 1))
            
            # Calculate points for the segment
            x1 = self.center[0] + self.radius * math.cos(angle1)
            y1 = self.center[1] + self.radius * math.sin(angle1)
            x2 = self.center[0] + self.radius * math.cos(angle2)
            y2 = self.center[1] + self.radius * math.sin(angle2)
            
            # Convert hue to RGB (saturation and value at 100%)
            rgb = hsv_to_rgb(i/360, 1, 1)
            color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
            
            # Draw the segment
            self.create_polygon(
                self.center[0], self.center[1],
                x1, y1,
                x2, y2,
                fill=color, outline=''
            )

    def draw_selection(self):
        """Draw the selection markers, connecting lines, and highlight."""
        self.delete('selection')
        
        # Draw connecting lines from center to markers
        for angle in [self.start_angle, self.end_angle]:
            rad = math.radians(angle)
            x = self.center[0] + self.radius * math.cos(rad)
            y = self.center[1] + self.radius * math.sin(rad)
            self.create_line(
                self.center[0], self.center[1],
                x, y,
                fill='black', width=2,
                tags=('selection', 'connecting_line')
            )
        
        # Draw the selection arc
        start = math.radians(self.start_angle)
        end = math.radians(self.end_angle)
        
        # Calculate midpoint
        mid_angle = (self.start_angle + self.end_angle) / 2
        if self.end_angle < self.start_angle:
            mid_angle = (self.start_angle + self.end_angle + 360) / 2
            if mid_angle >= 360:
                mid_angle -= 360
                
        # Draw connecting line for midpoint in a different style
        mid_rad = math.radians(mid_angle)
        mid_x = self.center[0] + self.radius * math.cos(mid_rad)
        mid_y = self.center[1] + self.radius * math.sin(mid_rad)
        self.create_line(
            self.center[0], self.center[1],
            mid_x, mid_y,
            fill='gray', width=2, dash=(5, 3),  # Dashed line for midpoint
            tags=('selection', 'connecting_line_mid')
        )
        
        # Draw selection indicators
        marker_radius = 8
        # Draw start and end markers
        for angle, tag in [(self.start_angle, 'start'), (self.end_angle, 'end')]:
            rad = math.radians(angle)
            x = self.center[0] + (self.radius + 5) * math.cos(rad)
            y = self.center[1] + (self.radius + 5) * math.sin(rad)
            
            self.create_oval(
                x - marker_radius, y - marker_radius,
                x + marker_radius, y + marker_radius,
                fill='white', outline='black',
                tags=('selection', f'marker_{tag}')
            )
        
        # Draw midpoint marker (slightly different style)
        rad = math.radians(mid_angle)
        x = self.center[0] + (self.radius + 5) * math.cos(rad)
        y = self.center[1] + (self.radius + 5) * math.sin(rad)
        
        self.create_oval(
            x - marker_radius, y - marker_radius,
            x + marker_radius, y + marker_radius,
            fill='gray', outline='black',
            tags=('selection', 'marker_mid')
        )

    def get_angle(self, x, y):
        """Convert coordinates to angle in degrees."""
        dx = x - self.center[0]
        dy = y - self.center[1]
        angle = math.degrees(math.atan2(dy, dx))
        return (angle + 360) % 360

    def normalize_angle(self, angle):
        """Normalize angle to 0-360 range."""
        return angle % 360

    def on_click(self, event):
        """Handle mouse click to start dragging."""
        x, y = event.x, event.y
        
        # Calculate positions of all markers
        start_rad = math.radians(self.start_angle)
        end_rad = math.radians(self.end_angle)
        mid_angle = (self.start_angle + self.end_angle) / 2
        if self.end_angle < self.start_angle:
            mid_angle = (self.start_angle + self.end_angle + 360) / 2
            if mid_angle >= 360:
                mid_angle -= 360
        mid_rad = math.radians(mid_angle)
        
        # Calculate marker positions
        start_x = self.center[0] + (self.radius + 5) * math.cos(start_rad)
        start_y = self.center[1] + (self.radius + 5) * math.sin(start_rad)
        end_x = self.center[0] + (self.radius + 5) * math.cos(end_rad)
        end_y = self.center[1] + (self.radius + 5) * math.sin(end_rad)
        mid_x = self.center[0] + (self.radius + 5) * math.cos(mid_rad)
        mid_y = self.center[1] + (self.radius + 5) * math.sin(mid_rad)
        
        # Calculate distances to markers
        start_dist = math.sqrt((x - start_x)**2 + (y - start_y)**2)
        end_dist = math.sqrt((x - end_x)**2 + (y - end_y)**2)
        mid_dist = math.sqrt((x - mid_x)**2 + (y - mid_y)**2)
        
        # Determine which marker to drag
        if start_dist < 10:
            self.dragging = 'start'
        elif end_dist < 10:
            self.dragging = 'end'
        elif mid_dist < 10:
            self.dragging = 'mid'
            # Store the initial range size for midpoint dragging
            if self.end_angle < self.start_angle:
                self.range_size = (360 - self.start_angle) + self.end_angle
            else:
                self.range_size = self.end_angle - self.start_angle

    def on_drag(self, event):
        """Handle dragging of selection markers."""
        if not self.dragging:
            return
            
        new_angle = self.get_angle(event.x, event.y)
        
        if self.dragging == 'start':
            self.start_angle = new_angle
        elif self.dragging == 'end':
            self.end_angle = new_angle
        elif self.dragging == 'mid':
            # Move both start and end while maintaining the range size
            half_range = self.range_size / 2
            self.start_angle = self.normalize_angle(new_angle - half_range)
            self.end_angle = self.normalize_angle(new_angle + half_range)
        
        self.draw_selection()
        
        if self.selection_callback:
            self.selection_callback(self.start_angle, self.end_angle)

    def on_right_click(self, event):
        """Handle right click to start range adjustment."""
        self.adjust_start_y = event.y
        self.initial_range = self.range_size

    def on_range_adjust(self, event):
        """Adjust the range size based on vertical mouse movement."""
        if not hasattr(self, 'adjust_start_y'):
            return
            
        # Calculate the midpoint
        mid_angle = (self.start_angle + self.end_angle) / 2
        if self.end_angle < self.start_angle:
            mid_angle = (self.start_angle + self.end_angle + 360) / 2
            if mid_angle >= 360:
                mid_angle -= 360
        
        # Adjust range based on vertical movement
        delta = (self.adjust_start_y - event.y) / 2  # Divided by 2 for more controlled adjustment
        new_range = max(0, min(360, self.initial_range + delta))
        
        # Update start and end angles
        half_range = new_range / 2
        self.start_angle = self.normalize_angle(mid_angle - half_range)
        self.end_angle = self.normalize_angle(mid_angle + half_range)
        self.range_size = new_range
        
        self.draw_selection()
        
        if self.selection_callback:
            self.selection_callback(self.start_angle, self.end_angle)

    def on_release(self, event):
        """Handle release of mouse button."""
        self.dragging = None
        if hasattr(self, 'adjust_start_y'):
            del self.adjust_start_y
            del self.initial_range

    def set_callback(self, callback):
        """Set the callback function for selection changes."""
        self.selection_callback = callback

    def get_selection(self):
        """Return the current selection range in degrees."""
        return self.start_angle, self.end_angle



class CurveEditor(tk.Frame):
    def __init__(self, parent, size=255, width=None, height=None, background='#333333', min_size=350, name=''):
        super().__init__(parent)
        
        self.callback = None
        self.size = size
        self.name = name
        self.min_size = min_size
        self.background = background
        self._width = width if width is not None else min_size
        self._height = height if height is not None else min_size
        
        self.points = [(0, 1), (1, 0)]
        self.selected_point = None
        self.drag_threshold = 10
        
        self.histogram_data = None
        self.histogram_bars = []
        
        self.canvas = tk.Canvas(
            self,
            bg=background,
            highlightthickness=0
        )
        self.canvas.pack(expand=True, fill='both')
        
        if width and height:
            self._width = width
            self._height = height
        else:
            self._width = min_size
            self._height = min_size
        
        self.canvas.config(width=self._width, height=self._height)
        
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<B1-Motion>', self.on_drag)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)
        self.canvas.bind('<Double-Button-1>', self.on_double_click)
        self.bind('<Configure>', self.on_resize)

        self.canvas.bind('<Button-3>', self.on_click_right)
    
        self.curve_line = None
        self.grid_lines = []
        


        self.smoothingFuncs = [
            lambda x,y : interpolate.CubicSpline(x, y,bc_type='natural'),
            lambda x,y : lambda p,x=x,y=y :np.interp(p,x,y),

            
        ]
        self.smoothingFuncIndex  = 0

        self.update_curve()
        
        self.callback = None

    def on_click_right(self, event):
        self.smoothingFuncIndex = (self.smoothingFuncIndex+1)%len(self.smoothingFuncs)
        self.update_curve()

    def setHistogram(self, values):

        if (self.histogram_data == values).all():
            return

        self.histogram_data = values
        
        self.canvas.delete("histogram")
        self.histogram_bars.clear()

        if values is None:
            return
            
        max_val = max(values)
        normalized = [v / max_val * (self.height * 0.8) for v in values]  # 80% of height
        
        total_bars = len(values)
        bar_width = self.width / total_bars
        
        for i, height in enumerate(normalized):
            x1 = int(i * bar_width)
            y1 = int(self.height)
            x2 = int((i + 1) * bar_width)
            y2 = int(self.height - height)
            
            bar = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill='#444444',
                outline='',
                tags='histogram',
            )
            self.histogram_bars.append(bar)
        
        for bar in self.histogram_bars:
            self.canvas.tag_lower(bar)
            
        self.update_curve()
        
    @property
    def width(self):
        return max(self._width, self.min_size)
    
    @property
    def height(self):
        return max(self._height, self.min_size)
    
    def _create_grid(self):

        for line in self.grid_lines:
            self.canvas.delete(line)
        self.grid_lines.clear()
        for i in range(5):  # 4 equal divisions
            x = i * (self.width / 4)
            y = i * (self.height / 4)
            
            # Vertical lines
            self.grid_lines.append(self.canvas.create_line(
                x, 0, x, self.height,
                fill='#444444', width=1
            ))
            
            # Horizontal lines
            self.grid_lines.append(self.canvas.create_line(
                0, y, self.width, y,
                fill='#444444', width=1
            ))
            self.canvas.create_text(25, 10, text=self.name, fill="grey")
            
    def _screen_to_norm(self, x, y):
        """Convert screen coordinates to normalized coordinates (0-1)"""
        return 1 - (x / self.width), 1 - (y / self.height)  # Note the flipped x coordinate
    
    def _norm_to_screen(self, x, y):
        """Convert normalized coordinates to screen coordinates"""
        return (1 - x) * self.width, (1 - y) * self.height  # Note the flipped x coordinate
    
    def on_resize(self, event):
        if event.width > 1 and event.height > 1:
            self._width = event.width
            self._height = event.height
            self.canvas.configure(width=self._width, height=self._height)
            self._create_grid()
            
            # Redraw histogram if exists
            if self.histogram_data:
                self.setHistogram(self.histogram_data)
                
            self.update_curve()
            self.update_idletasks()
            self.canvas.scale("all", 0, 0, 
                            event.width / max(1, self.canvas.winfo_reqwidth()),
                            event.height / max(1, self.canvas.winfo_reqheight()))
    
    def update_curve(self):
        # Sort points by x coordinate
        self.points.sort(key=lambda p: p[0])
        
        # Create interpolation points
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])
        
        # Create a dense set of points for smooth curve
        t = np.linspace(0, 1, 100)
        
        try:
            # Create cubic spline interpolation
            cs = self.smoothingFuncs[self.smoothingFuncIndex](x,y)
            interpolated = cs(t)
            
            # Convert to screen coordinates
            screen_points = [self._norm_to_screen(t[i], interpolated[i]) 
                           for i in range(len(t))]
            
            # Delete old curve if it exists
            self.canvas.delete("curveline")
            
            # Create new curve
            self.curve_line = self.canvas.create_line(
                screen_points,
                fill='white',
                width=2,
                smooth=True,
                tags="curveline"
            )
            
            # Draw control points
            self.draw_points()
            
            # Generate and notify about new LUT
            self.generate_lut()
            
        except ValueError:
            # Handle case where spline interpolation fails
            pass
    
    def draw_points(self):
        # Clear existing points
        self.canvas.delete("pointmarker")
        
        # Draw new points
        for x, y in self.points:
            screen_x, screen_y = self._norm_to_screen(x, y)
            
            # White outline
            self.canvas.create_oval(
                screen_x - 5, screen_y - 5,
                screen_x + 5, screen_y + 5,
                fill='white',
                outline='white',
                tags="pointmarker"
            )
            # Black center
            self.canvas.create_oval(
                screen_x - 3, screen_y - 3,
                screen_x + 3, screen_y + 3,
                fill='black',
                outline='black',
                tags="pointmarker"
            )
    
    def on_click(self, event):
        for i, (x, y) in enumerate(self.points):
            screen_x, screen_y = self._norm_to_screen(x, y)
            if abs(event.x - screen_x) < self.drag_threshold and \
               abs(event.y - screen_y) < self.drag_threshold:
                self.selected_point = i
                return

        # If not near existing point, add new point
        new_x, new_y = self._screen_to_norm(event.x, event.y)
        newpoint = (new_x, new_y)
        self.points.append(newpoint)
        self.update_curve()
        self.selected_point = self.points.index(newpoint)

    def on_mousewheel(self, event):
        direction = event.delta

        print(self.points)

        if direction>0:
            delta = -0.008
        elif direction<0:
            delta = 0.008
        else:
            return

        for i,(px,py) in enumerate(self.points[1:-1]):

            prev_x = self.points[(i+1) - 1][0]
            next_x = self.points[(i+1) + 1][0]
            nx = max(prev_x+0.001, min(px+delta, next_x-0.001))

            self.points[i+1] = (nx, py)
        
        self.update_curve()


    def on_drag(self, event):
        if self.selected_point is not None:
            # Convert screen coordinates to normalized
            x, y = self._screen_to_norm(event.x, event.y)
            
            # Constrain x movement for endpoint points
            if self.selected_point == 0:
                x = 0
            elif self.selected_point == len(self.points) - 1:
                x = 1
            else:
                # Constrain x to be between neighboring points
                prev_x = self.points[self.selected_point - 1][0]
                next_x = self.points[self.selected_point + 1][0]
                x = max(prev_x, min(x, next_x))
            
            # Constrain y to 0-1 range
            y = max(0, min(y, 1))
            
            # Update point position
            self.points[self.selected_point] = (x, y)
            self.update_curve()
    
    def on_release(self, event):
        self.selected_point = None
    
    def on_double_click(self, event):
        # Check if clicked near existing point
        for i, (x, y) in enumerate(self.points):
            screen_x, screen_y = self._norm_to_screen(x, y)
            if abs(event.x - screen_x) < self.drag_threshold and \
               abs(event.y - screen_y) < self.drag_threshold:
                # Don't remove endpoint points
                if i != 0 and i != len(self.points) - 1:
                    self.points.pop(i)
                    self.update_curve()
                return
        
        # If not near existing point, add new point
        new_x, new_y = self._screen_to_norm(event.x, event.y)
        self.points.append((new_x, new_y))
        self.update_curve()
    
    def generate_lut(self):
        """Generate LUT arrays based on current curve"""
        # Create input array
        lut_in = np.linspace(0, 1, self.size)
        
        # Sort points and create interpolation
        self.points.sort(key=lambda p: p[0])
        x = np.array([p[0] for p in self.points])
        y = np.array([p[1] for p in self.points])
        
        try:
            # Create cubic spline interpolation
            cs = self.smoothingFuncs[self.smoothingFuncIndex](x,y)
            # Generate output values
            lut_out = cs(lut_in)
            # Clip to 0-1 range
            lut_out = np.clip(lut_out, 0, 1)
            
            # Call callback if exists
            if self.callback:
                self.callback(lut_in, lut_out)
            
            return lut_in, lut_out
            
        except ValueError:
            return None
    
    def set_callback(self, callback):
        self.callback = callback
    
    def reset(self):
        self.points = [(0, 1), (1, 0)]
        self.update_curve()

class Controller():
    def __init__(self, master=None):
        self.master = master
        self.monitors = screeninfo.get_monitors()
        
        self.screenPadding = 160
        self.webcamIndex = 0

        self.monitorIndex = 0
        self.projectorIndex = -1

        self.monitor   = self.monitors[self.monitorIndex]
        self.projector = self.monitors[self.projectorIndex]       

        self.points=[]
        self.corners=[]

        self.zoom_var = tk.DoubleVar(value=1.0)
        self.blur_var = tk.IntVar(value=0)
        self.posterize_var = tk.IntVar(value=0)

        self.canvas_mix_var = tk.IntVar(value=0)
        self.original_mix_var = tk.IntVar(value=0)
        self.overlay_mix_var = tk.IntVar(value=0)

        self.sample_frames_var = tk.IntVar(value=3)

        self.exponent_var = tk.DoubleVar(value=1)
        self.threshold_var = tk.DoubleVar(value=0)

        self.grey_enabled   = tk.BooleanVar(value=True)
        self.invert_var = tk.BooleanVar(value=False)
        self.normalize_var = tk.BooleanVar(value=True)
        self.invert_diff_var = tk.BooleanVar(value=False)
        self.hue_filter = tk.BooleanVar(value=False)


        self.slic_level = tk.IntVar(value=0)
        self.slic_iterations = tk.IntVar(value=40)
        self.mode_var = tk.StringVar(value='Light+Dark')
        self.loss_var = tk.StringVar(value='Multiscale loss l3')
        self.width_var = tk.IntVar(value=210)
        self.height_var = tk.IntVar(value=297)

        self.loop_delay_var = tk.IntVar(value=60)

        self.ignore_border_var = tk.IntVar(value=0)

        main_frame = ttk.Frame(master, padding="5")
        main_frame.pack(expand=True, fill='both')

        controls = ttk.Frame(main_frame, padding="5")
        controls.pack(fill='x', padx=5)

        row = 0

        ttk.Label(controls, text="Actual Width").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(controls, textvariable=self.width_var).grid(row=row, column=1, sticky='ew', padx=5)
        
        row += 1
        ttk.Label(controls, text="Actual Height").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(controls, textvariable=self.height_var).grid(row=row, column=1, sticky='ew', padx=5)

        row += 1
        ttk.Label(controls, text="SLIC Level").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(controls, textvariable=self.slic_level).grid(row=row, column=1, sticky='ew', padx=5)

        row += 1
        ttk.Label(controls, text="SLIC Iterations").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(controls, textvariable=self.slic_iterations).grid(row=row, column=1, sticky='ew', padx=5)


        row += 1
        ttk.Label(controls, text="Ignore Border Pixels").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(controls, textvariable=self.ignore_border_var).grid(row=row, column=1, sticky='ew', padx=5)

        row += 1
        ttk.Label(controls, text="Loop Delay").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(controls, textvariable=self.loop_delay_var).grid(row=row, column=1, sticky='ew', padx=5)

        row += 1
        ttk.Label(controls, text="Sample Frames").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Entry(controls, textvariable=self.sample_frames_var).grid(row=row, column=1, sticky='ew', padx=5)

        def update_zoom_label(*args):
            zoom_label.configure(text=f"{float(self.zoom_var.get()):.3f}")
            
        def update_blur_label(*args):
            blur_label.configure(text=f"{float(self.blur_var.get()):.3f}")

        def update_canvas_mix_label(*args):
            canvas_mix_label.configure(text=f"{float(self.canvas_mix_var.get()):.3f}")

        def update_original_mix_label(*args):
            original_mix_label.configure(text=f"{float(self.original_mix_var.get()):.3f}")

        def update_exponent_label(*args):
            exponent_label.configure(text=f"{float(self.exponent_var.get()):.3f}")


        def update_threshold_label(*args):
            threshold_label.configure(text=f"{float(self.threshold_var.get()):.3f}")

        def update_overlay_mix_label(*args):
            overlay_mix_label.configure(text=f"{float(self.overlay_mix_var.get()):.3f}")

        def update_posterize_label(*args):
            posterize_label.configure(text=f"{float(self.posterize_var.get()):.3f}")

        def update_duotone_threshold_label(*args):
            duotone_threshold_label.configure(text=f"{float(self.duotone_threshold_var.get()):.3f}")

        def recaculate_on_change(*args):
            self.dorecalc()

        row += 1
        ttk.Label(controls, text="Zoom").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Scale(controls, from_=0.01, to=4.0, variable=self.zoom_var, orient='horizontal'
                 ).grid(row=row, column=1, sticky='ew', padx=5)
        zoom_label = ttk.Label(controls, text=f"{float(self.zoom_var.get()):.3f}")
        zoom_label.grid(row=row, column=2, padx=5)
        self.zoom_var.trace('w', update_zoom_label)
        self.zoom_var.trace('w', recaculate_on_change)

        row += 1
        ttk.Label(controls, text="Blur").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Scale(controls, from_=0, to=200, variable=self.blur_var, orient='horizontal'
                 ).grid(row=row, column=1, sticky='ew', padx=5)
        blur_label = ttk.Label(controls, text=f"{float(self.blur_var.get()):.3f}")
        blur_label.grid(row=row, column=2, padx=5)
        self.blur_var.trace('w', update_blur_label)
        self.blur_var.trace('w', recaculate_on_change)


        row += 1
        ttk.Label(controls, text="Canvas mix").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Scale(controls, from_=0, to=100, variable=self.canvas_mix_var, orient='horizontal'
                 ).grid(row=row, column=1, sticky='ew', padx=5)
        canvas_mix_label = ttk.Label(controls, text=f"{float(self.canvas_mix_var.get()):.3f}")
        canvas_mix_label.grid(row=row, column=2, padx=5)
        self.canvas_mix_var.trace('w', update_canvas_mix_label)


        row += 1
        ttk.Label(controls, text="Original mix").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Scale(controls, from_=0, to=100, variable=self.original_mix_var, orient='horizontal'
                 ).grid(row=row, column=1, sticky='ew', padx=5)
        original_mix_label = ttk.Label(controls, text=f"{float(self.original_mix_var.get()):.3f}")
        original_mix_label.grid(row=row, column=2, padx=5)
        self.original_mix_var.trace('w', update_original_mix_label)

        row += 1
        ttk.Label(controls, text="Overlay mix").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Scale(controls, from_=0, to=100, variable=self.overlay_mix_var, orient='horizontal'
                 ).grid(row=row, column=1, sticky='ew', padx=5)
        overlay_mix_label = ttk.Label(controls, text=f"{float(self.overlay_mix_var.get()):.3f}")
        overlay_mix_label.grid(row=row, column=2, padx=5)
        self.overlay_mix_var.trace('w', update_overlay_mix_label)


        row += 1
        ttk.Label(controls, text="Exponent").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Scale(controls, from_=1, to=64, variable=self.exponent_var, orient='horizontal'
                 ).grid(row=row, column=1, sticky='ew', padx=5)
        exponent_label = ttk.Label(controls, text=f"{float(self.exponent_var.get()):.3f}")
        exponent_label.grid(row=row, column=2, padx=5)
        self.exponent_var.trace('w', update_exponent_label)

        row += 1
        ttk.Label(controls, text="Threshold").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Scale(controls, from_=0, to=255, variable=self.threshold_var, orient='horizontal'
                 ).grid(row=row, column=1, sticky='ew', padx=5)
        threshold_label = ttk.Label(controls, text=f"{float(self.threshold_var.get()):.3f}")
        threshold_label.grid(row=row, column=2, padx=5)
        self.threshold_var.trace('w', update_threshold_label)


        row += 1
        ttk.Label(controls, text="Posterize").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Scale(controls, from_=0, to=25, variable=self.posterize_var, orient='horizontal'
                 ).grid(row=row, column=1, sticky='ew', padx=5)
        posterize_label = ttk.Label(controls, text=f"{float(self.posterize_var.get()):.3f}")
        posterize_label.grid(row=row, column=2, padx=5)
        self.posterize_var.trace('w', update_posterize_label)

        row += 1

        check_frame = ttk.Frame(controls)
        check_frame.grid(row=row, column=0, columnspan=3, sticky='w', pady=2)


        ttk.Checkbutton(check_frame, text="Luminance Only", variable=self.grey_enabled).pack(side='left', padx=5)
        ttk.Checkbutton(check_frame, text="Invert Guidance", variable=self.invert_var).pack(side='left', padx=5)
        
        ttk.Checkbutton(check_frame, text="Invert Differences", variable=self.invert_diff_var).pack(side='left', padx=5)

        ttk.Checkbutton(check_frame, text="Normalize Levels", variable=self.normalize_var).pack(side='left', padx=5)


        ttk.Checkbutton(check_frame, text="Filter Hues", variable=self.hue_filter).pack(side='left', padx=5)

        self.grey_enabled.trace('w', recaculate_on_change)
        self.invert_var.trace('w', recaculate_on_change)
        self.invert_diff_var.trace('w', recaculate_on_change)
        self.normalize_var.trace('w', recaculate_on_change)
        self.hue_filter.trace('w', recaculate_on_change)



        row += 1
        ttk.Label(controls, text="Mode").grid(row=row, column=0, sticky='w', pady=2)
        ttk.OptionMenu(controls, self.mode_var, self.mode_var.get(),
                     'Light+Dark', 'Light', 'Dark', 'Original', 'Canvas'
                     ).grid(row=row, column=1, sticky='ew', padx=5)
        self.mode_var.trace('w', recaculate_on_change)

        row += 1
        ttk.Label(controls, text="Loss Type").grid(row=row, column=0, sticky='w', pady=2)
        ttk.OptionMenu(controls, self.loss_var, self.loss_var.get(),
                     *losses.keys()
                     ).grid(row=row, column=1, sticky='ew', padx=5)
        self.loss_var.trace('w', recaculate_on_change)

        row += 1

        self.primary_color    = list((255,0,0))
        self.secondary_color  = list((0,0,255))

        def pick_primary():
           color = colorchooser.askcolor(title="Choose primary (darken)")[0]
           if color:
               self.primary_color = list(color)

        def pick_secondary(): 
           color = colorchooser.askcolor(title="Choose secondary (lighten)")[0]
           if color:
               self.secondary_color = list(color)


        ttk.Label(controls, text="Guidance Colours").grid(row=row, column=0, sticky='w', pady=2)
        ttk.Button(controls, text="Pick Primary (darken)", command=pick_primary).grid(row=row, column=1, sticky='ew', padx=5)
        ttk.Button(controls, text="Pick Secondary (lighten)", command=pick_secondary).grid(row=row, column=2, sticky='ew', padx=5)




        self.notebook = ttk.Notebook(main_frame)



        self.editor_target = CurveEditor(main_frame,name='Original')
        self.editor_target.pack(fill='both', expand=True)
        self.editor_target.set_callback(self.lutChange_target)
        self.notebook.add(self.editor_target,text='Original Curves')

        self.editor_canvas = CurveEditor(main_frame,name='Canvas')
        self.editor_canvas.pack(fill='both', expand=False)
        self.editor_canvas.set_callback(self.lutChange_canvas)
        self.notebook.add(self.editor_canvas,text='Canvas Curves')

        self.editor_out = CurveEditor(main_frame,name='Output')
        self.editor_out.pack(fill='both', expand=True)
        self.editor_out.set_callback(self.lutChange_out)
        self.notebook.add(self.editor_out,text='Output Curves')

        self.selector = HueRangeSelector(main_frame)
        self.selector.set_callback(self.on_hue_selection_change)
        self.selector.pack(padx=20, pady=10, fill='x', expand=True)

        self.notebook.add(self.selector,text='Hue Filtering')
        self.notebook.pack(fill='both', expand=True)


        self.lut_canvas = None
        self.lut_target = None
        self.lut_out = None


        self.progressbar = ttk.Progressbar(main_frame)
        self.progressbar['mode'] = "determinate"
        self.progressbar.pack(fill='both', expand=True, padx=2)


        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill='x', pady=2)

        ttk.Button(btn_frame, text="Load Image", command=self.loadImage).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Register Devices", command=self.register).pack(side='left', padx=2)

        ttk.Button(btn_frame, text="Capture base image", command=self.setBaseDiff).pack(side='left', padx=2)
        ttk.Button(btn_frame, text="Clear base image", command=self.clearBaseDiff).pack(side='left', padx=2)

        ttk.Button(btn_frame, text="Recalculate", command=self.dorecalc).pack(side='left', padx=2)

        self.recalcloopButton = ttk.Button(btn_frame, text="Recalc Loop", command=self.dorecalcLoop)
        self.recalcloopButton.pack(side='left', padx=5)

        # Buttons frame
        btn_frame2 = ttk.Frame(main_frame)
        btn_frame2.pack(fill='x', pady=2)

        ttk.Button(btn_frame2, text="Canvas Range to Original", command=self.canvasRangeToOriginal).pack(side='left', padx=2)
        ttk.Button(btn_frame2, text="Original Range to Canvas", command=self.originalRangeToCanvas).pack(side='left', padx=2)


        controls.columnconfigure(1, weight=1)

        self.cap = cv2.VideoCapture(self.webcamIndex, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        #self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))



        self.thread = None
        self.recalc = True
        
        self.last_mx=0
        self.last_my=0

        self.defaultFile = None
        self.regdata = None
        
        self.baseDiff = None
        self.projected_grey = None

        self.recalcloop = False
        self.lastRecalc = time.time()

        self.camroll = (0,0)

        self.master.attributes('-topmost', 1)

        self.master.after(200, lambda:self.loadImage(withCrop=False))

        self.hue_start_angle = 0
        self.hue_end_angle = 0

    def extract_hue(self, img, hue_min, hue_max):

        print('raw',hue_min,hue_max)

        hue_min = int((hue_min % 360) * 179 / 360)
        hue_max = int((hue_max % 360) * 179 / 360)

        print('cv2',hue_min,hue_max)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_bound_low = np.array([0, 0, 0])
        upper_bound_low = np.array([255, 11, 11])
        masklowhue  = cv2.inRange(hsv, lower_bound_low, upper_bound_low)

        hsv = cv2.GaussianBlur(hsv,(7,7),0)

        if hue_min <= hue_max:
            lower_bound = np.array([hue_min, 0, 0])
            upper_bound = np.array([hue_max, 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
        else:
            lower_bound1 = np.array([0, 0, 0])
            upper_bound1 = np.array([hue_max, 255, 255])
            mask1 = cv2.inRange(hsv, lower_bound1, upper_bound1)
            
            lower_bound2 = np.array([hue_min, 0, 0])
            upper_bound2 = np.array([255, 255, 255])
            mask2 = cv2.inRange(hsv, lower_bound2, upper_bound2)
            
            mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.bitwise_or(mask, masklowhue)

        result = cv2.bitwise_and(img, img, mask=mask)

        cv2.imshow('extract_hue',result)
        cv2.waitKey(1)
        return result

    def on_hue_selection_change(self, start_angle, end_angle):
        self.hue_start_angle = start_angle
        self.hue_end_angle = end_angle
        self.recalc = True

    def canvasRangeToOriginal(self):
        b = cv2.GaussianBlur(self.norm_target_img,(7,7),0)
        a = cv2.GaussianBlur(self.norm_current,(7,7),0)
        scale = create_blackwhite_lut(a,b)
        self.editor_canvas.reset()
        self.editor_target.reset()
        self.editor_target.points = [(0, scale[-1]/255), (1,scale[0]/255)]
        self.editor_target.update_curve()

    def originalRangeToCanvas(self):
        b = cv2.GaussianBlur(self.norm_target_img,(7,7),0)
        a = cv2.GaussianBlur(self.norm_current,(7,7),0)
        scale = create_blackwhite_lut(b,a)
        self.editor_target.reset()
        self.editor_canvas.reset()
        self.editor_canvas.points = [(0, scale[-1]/255), (1,scale[0]/255)]
        self.editor_canvas.update_curve()

    def lutChange_canvas(self,lut_in,lut_out):
        self.lut_canvas = (np.interp(np.arange(0, 256), lut_in*255, lut_out[::-1]*255))
        self.recalc = True

    def lutChange_target(self,lut_in,lut_out):
        self.lut_target = (np.interp(np.arange(0, 256), lut_in*255, lut_out[::-1]*255))
        self.recalc = True

    def lutChange_out(self,lut_in,lut_out):
        self.lut_out = (np.interp(np.arange(0, 256), lut_in*255, lut_out[::-1]*255))


    def dorecalcLoop(self):
        self.recalcloop = not self.recalcloop
        if self.recalcloop:
            self.recalcloopButton.config(text="Stop Recalc Loop")
        else:
            self.recalcloopButton.config(text='Recalc Loop')


    def setBaseDiff(self):
        if self.projected_grey is not None:
            cv2.imshow('projector',self.projected_grey)
            cv2.waitKey(250)
            current = self.getframe(samples=20)
            warped = cv2.warpPerspective(current, self.matrix, (self.target_h,self.target_w),flags=cv2.INTER_AREA)
            warped = cv2.GaussianBlur(warped,(7,7),0)

            self.baseDiff = warped.astype(float)
            self.recalc = True

    def clearBaseDiff(self):
        self.baseDiff = None
        self.recalc = True



    def register(self):
        self.points = []
        self.canvas = []
        if self.target is not None:
            self.regdata = None
            self.registerProjector()

    def loadImage(self,withCrop=True):
        if self.defaultFile is not None:
            self.progressbar['mode'] = "indeterminate"
            self.progressbar.start(20)
            im = cv2.imread(self.defaultFile)
            self.target = im
            self.registerProjector(withCrop=withCrop)
        else:
            options = dict(parent=self.master)
            fn = filedialog.askopenfilename(**options)
            im = cv2.imread(fn)
            if im is not None:
                self.progressbar['mode'] = "indeterminate"
                self.progressbar.start(20)
                self.target = im
                self.registerProjector(withCrop=withCrop)

    def dorecalc(self):
        self.recalc = True

    def select_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            param.append((x, y))
        self.last_mx=x
        self.last_my=y


    def getframe(self,samples=None):
        if samples is None:
            samples = self.sample_frames_var.get()
        frames = []
        self.cap.read()
        for _ in range(samples):
            ret, frame = self.cap.read()
            frames.append(frame.astype(np.float32))
        median_frame = np.median(frames, axis=0).astype(np.uint8)
        return median_frame

    def _capture_loop(self):
        cv2.imshow('projector',self.projected_grey)
        iteration=0

        basePathForSave = 'savedSessions' 
        os.path.exists(basePathForSave) or os.mkdir(basePathForSave)
        basePathForSave = os.path.join(basePathForSave,datetime.datetime.now().isoformat().replace(':','-'))
        os.path.exists(basePathForSave) or os.mkdir(basePathForSave)

        diff_lighten = np.zeros_like(self.target_img)
        diff_darken = np.zeros_like(self.target_img)
        diff_image = np.zeros_like(self.target_img)

        while True:
            lastWasRecalc=False

            try:
                iteration+=1
                if self.recalc:
                    startts=time.time()
                    lastWasRecalc=True
                    self.lastRecalc = time.time()
                    self.progressbar['mode'] = "indeterminate"
                    self.progressbar.start(20)
                    cv2.imshow('projector',self.projected_grey)
                    cv2.waitKey(1350)
                    current = self.getframe()
                    cv2.waitKey(1350)
                    warped = cv2.warpPerspective(current, self.matrix, (self.target_h,self.target_w),flags=cv2.INTER_AREA)
                    bg_warped = warped

                    cv2.imwrite(os.path.join(basePathForSave,f"{iteration:05}.png"),current)


                    target_initial = self.target_img

                    if self.hue_filter.get():
                        target_initial = self.extract_hue(target_initial, self.hue_start_angle, self.hue_end_angle)

                    if self.normalize_var.get():
                        norm_target_img = cv2.normalize(target_initial, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F )
                    else:
                        norm_target_img = target_initial.astype(np.float32)


                    if self.baseDiff is not None:
                        warped = np.clip(warped.astype(float)+(255-self.baseDiff.astype(float)),0,255)

                    if self.normalize_var.get():
                        norm_current  = cv2.normalize(warped,  None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F )
                    else:
                        norm_current = warped.astype(np.float32)

                    norm_target_img_shape = norm_target_img.shape

                    if self.zoom_var.get() != 1:
                        zoom_factor = self.zoom_var.get()
                        height, width = norm_target_img.shape[:2]
                        
                        if zoom_factor < 1:  # Zooming out
                            # Calculate new dimensions
                            w_scaled = int(width * zoom_factor)
                            h_scaled = int(height * zoom_factor)
                            
                            # Resize image to smaller size
                            resized = cv2.resize(norm_target_img, (w_scaled, h_scaled), interpolation=cv2.INTER_CUBIC)
                            
                            # Create padded image
                            padded = np.zeros_like(norm_target_img)
                            x_offset = (width - w_scaled) // 2
                            y_offset = (height - h_scaled) // 2
                            padded[y_offset:y_offset+h_scaled, x_offset:x_offset+w_scaled] = resized
                            
                            norm_target_img = padded
                        else:  # Zooming in (original code)
                            center_x, center_y = width/2, height/2
                            w_scaled = int(width/zoom_factor)
                            h_scaled = int(height/zoom_factor)
                            left = int(center_x - w_scaled/2)
                            top = int(center_y - h_scaled/2)
                            roi_left = max(0, left)
                            roi_top = max(0, top)
                            roi_right = min(width, left + w_scaled)
                            roi_bottom = min(height, top + h_scaled)
                            roi = norm_target_img[roi_top:roi_bottom, roi_left:roi_right]
                            norm_target_img = cv2.resize(roi, (width, height), interpolation=cv2.INTER_CUBIC)
                            
                        norm_target_img_shape = norm_target_img.shape


                    if self.slic_level.get() != 0:

                        norm_target_img = cv2.GaussianBlur(norm_target_img,(3,3),0)

                        slic = cv2.ximgproc.createSuperpixelSLIC(norm_target_img, algorithm=cv2.ximgproc.MSLIC, region_size=self.slic_level.get())
                        slic.iterate(self.slic_iterations.get())
                        labels = slic.getLabels()

                        # Create blocky effect
                        result = norm_target_img.copy()
                        for label in np.unique(labels):
                            mask = labels == label
                            # Average color for each superpixel
                            color = np.mean(norm_target_img[mask], axis=0)
                            result[mask] = color

                        norm_target_img = result


                    if self.grey_enabled.get():
                        norm_target_img = RGB2GreyCustom(norm_target_img)
                        norm_current    = RGB2GreyCustom(norm_current)
                    

                    if self.blur_var.get() > 0:
                        ksize = (self.blur_var.get(), self.blur_var.get()) 
                        norm_target_img = cv2.blur(norm_target_img, ksize, cv2.BORDER_DEFAULT) 



                    self.norm_current = norm_current
                    if self.lut_canvas is not None:
                        norm_current = cv2.LUT(norm_current.astype(np.uint8), self.lut_canvas).astype(np.float32)
                    hist = norm_current
                    hist = cv2.calcHist([hist], [0], None, [127], [0, 256])
                    self.editor_canvas.setHistogram(hist)




                    self.norm_target_img = norm_target_img
                    if self.lut_target is not None:
                        norm_target_img = cv2.LUT(norm_target_img.astype(np.uint8), self.lut_target).astype(np.float32)
                    hist = norm_target_img
                    hist = cv2.calcHist([hist], [0], None, [127], [0, 256])
                    self.editor_target.setHistogram(hist)


                    mode = self.mode_var.get()

                    previewMode = False
                    if mode == 'Original':
                        if len(norm_target_img.shape) == 2:
                            diff_image = cv2.cvtColor(norm_target_img, cv2.COLOR_GRAY2BGR )
                        else:
                            diff_image = norm_target_img
                        previewMode = True

                    if mode == 'Canvas':
                        if len(norm_current.shape) == 2:
                            diff_image = cv2.cvtColor(norm_current, cv2.COLOR_GRAY2BGR )
                        else:
                            diff_image = norm_current
                        previewMode = True

                    if previewMode:
                        diff_image = self.resize_to_smaller(diff_image, self.monitor.width-self.screenPadding, self.monitor.height-self.screenPadding)
                        cv2.imshow('diff image',diff_image.astype(np.uint8))
                        cv2.waitKey(1)
                        self.recalc=False
                        self.progressbar.stop()
                        self.progressbar['mode'] = "determinate"
                        self.progressbar.step(99.9)
                        continue

                    diff = losses[self.loss_var.get()](norm_current ,norm_target_img)
                    if self.invert_diff_var.get():
                        diff = -diff

                    if self.ignore_border_var.get()>0:
                        n = self.ignore_border_var.get()
                        diff[:n, :] = 0  
                        diff[-n:, :] = 0
                        diff[:, :n] = 0
                        diff[:, -n:] = 0

                    diff_darken = (cv2.convertScaleAbs(cv2.max(diff,0)))
                    diff_darken = cv2.normalize(diff_darken, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
                    diff_darken_base = diff_darken.astype(np.uint8)

                    diff_lighten = (cv2.convertScaleAbs(cv2.min(diff,0)))
                    diff_lighten = cv2.normalize(diff_lighten, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F).astype(np.uint8)
                    diff_lighten_base = diff_lighten.astype(np.uint8)

                    diff_darken = diff_darken_base.copy()
                    diff_lighten = diff_lighten_base.copy()

                    self.recalc = False

                    if diff_darken.ndim == 2:
                        diff_darken = cv2.cvtColor(diff_darken, cv2.COLOR_GRAY2BGR )
                        diff_lighten = cv2.cvtColor(diff_lighten, cv2.COLOR_GRAY2BGR )



                colour1 = cv2.cvtColor(np.array([[self.primary_color]]).astype(np.uint8), cv2.COLOR_RGB2HLS).astype(np.float32)
                colour2 = cv2.cvtColor(np.array([[self.secondary_color]]).astype(np.uint8), cv2.COLOR_RGB2HLS).astype(np.float32)

                diff_lighten = diff_lighten.astype(np.float32)
                diff_darken = diff_darken.astype(np.float32)
                diff_image = np.zeros_like(diff_lighten)




                alphaChannel = np.full_like(diff_image,255)

                if mode in ('Light+Dark','Dark','Light'):

                    darkenchannel  = diff_darken[:,:,0].copy()
                    lightenchannel = diff_lighten[:,:,0].copy()


                    if self.posterize_var.get() > 0:
                        darkenchannel = posterize(darkenchannel,self.posterize_var.get())
                        lightenchannel = posterize(lightenchannel,self.posterize_var.get())

                    if self.exponent_var.get() != 1:
                        if 'Dark' in mode:
                            darkenchannel = np.log(darkenchannel.astype(np.float64) + 1)
                            darkenchannel = darkenchannel**self.exponent_var.get()
                            darkenchannel = cv2.normalize(darkenchannel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)                     

                        if 'Light' in mode:
                            lightenchannel = np.log(lightenchannel.astype(np.float64) + 1)
                            lightenchannel = lightenchannel**self.exponent_var.get()
                            lightenchannel = cv2.normalize(lightenchannel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


                    if 'Dark' not in mode:
                        darkenchannel = np.zeros_like(darkenchannel)

                    if 'Light' not in mode:
                        lightenchannel = np.zeros_like(lightenchannel)

                    darken_mask = (darkenchannel > 0)
                    lighten_mask = (lightenchannel > 0)

                    diff_image[:,:,1] = lightenchannel + darkenchannel

                    diff_image[:,:,1] = np.clip(diff_image[:,:,1],0,255)

                    diff_image[:,:,0][darken_mask]   = colour1[0][0][0]
                    diff_image[:,:,2][darken_mask]   = colour1[0][0][2]

                    diff_image[:,:,0][lighten_mask]  = colour2[0][0][0]
                    diff_image[:,:,2][lighten_mask]  = colour1[0][0][2]

                    diff_image[:,:,1] = np.clip(diff_image[:,:,1],0,255)

                    if self.threshold_var.get() > 0:
                        diff_image[:,:,1] = cv2.threshold(diff_image[:,:,1].astype(np.uint8),self.threshold_var.get(),255,cv2.THRESH_TOZERO)[1].astype(np.float32)

                    alphaChannel = diff_image[:,:,1].copy()

                    if self.invert_var.get():
                        diff_image[:,:,1] = 255-diff_image[:,:,1]

                    diff_image = (cv2.cvtColor(diff_image.astype(np.uint8), cv2.COLOR_HLS2BGR))


                if mode == 'Original':
                    if len(norm_target_img.shape) == 2:
                        diff_image = cv2.cvtColor(norm_target_img, cv2.COLOR_GRAY2BGR )
                    else:
                        diff_image = norm_target_img
                if mode == 'Canvas':
                    if len(norm_current.shape) == 2:
                        diff_image = cv2.cvtColor(norm_current, cv2.COLOR_GRAY2BGR )
                    else:
                        diff_image = norm_current

                diff_image = diff_image.astype(np.uint8)


                if self.lut_out is not None:
                    diff_image = cv2.LUT(diff_image.astype(np.uint8), self.lut_out).astype(np.uint8)
                hist = cv2.cvtColor(diff_image, cv2.COLOR_BGR2GRAY)

                hist = cv2.calcHist([hist], [0], None, [127], [0, 256])
                self.editor_out.setHistogram(hist)

                if self.canvas_mix_var.get() > 0:
                    alpha = self.canvas_mix_var.get() / 100
                    beta = 1 - alpha
                    diff_image = cv2.addWeighted(warped, alpha, diff_image, beta, 0)


                if self.original_mix_var.get() > 0:
                    alpha = self.original_mix_var.get() / 100
                    beta = 1 - alpha

                    if len(norm_target_img.shape) == 2:
                        targetmix = cv2.cvtColor(norm_target_img, cv2.COLOR_GRAY2BGR ).astype(np.uint8)
                    else:
                        targetmix = norm_target_img.astype(np.uint8)

                    diff_image = cv2.addWeighted(targetmix, alpha, diff_image, beta, 0)






                if self.overlay_mix_var.get() > 0:

                    if len(bg_warped.shape) == 2:
                        background = cv2.cvtColor(bg_warped, cv2.COLOR_GRAY2BGR ).astype(float)/255.0
                    else:
                        background = bg_warped.astype(float)/255.0

                    if len(diff_image.shape) == 2:
                        foreground_o = cv2.cvtColor(diff_image, cv2.COLOR_GRAY2BGR ).astype(float)/255.0
                    else:
                        foreground_o = diff_image.astype(float)/255.0

                    if len(alphaChannel.shape) == 2:
                        alpha = cv2.cvtColor(alphaChannel, cv2.COLOR_GRAY2BGR ).astype(float)/255.0
                    else:
                        alpha = alphaChannel.astype(float)/255.0

                    # Multiply the foreground with the alpha matte
                    foreground = cv2.multiply(alpha, foreground_o)

                    # Multiply the background with ( 1 - alpha )
                    background = cv2.multiply(1.0 - alpha, background)

                    # Add the masked foreground and background.
                    diff_image = cv2.add(foreground, background)

                    diff_image = cv2.addWeighted(foreground_o, 1.0 - (self.overlay_mix_var.get()/100), diff_image, (self.overlay_mix_var.get()/100), 0)

                if lastWasRecalc:
                    self.lastRecalc = time.time()
                if self.recalcloop:
                    maxtime = self.loop_delay_var.get()
                    timesincelast = time.time()-self.lastRecalc
                    if timesincelast>maxtime:
                        self.recalc=True
                    else:
                        pcw = int(min((timesincelast/maxtime)*diff_image.shape[1],diff_image.shape[1]))
                        
                        diff_image[:2,0:pcw,:] = 255-diff_image[:2,0:pcw,:]

                        if timesincelast/maxtime > 0.9 and iteration%2==0:
                            diff_image[:,:] = (0,0,100) 



                projected = cv2.warpPerspective(
                    diff_image,
                    self.projectorMatrix,
                    (self.surface_h,self.surface_w),
                    flags=cv2.INTER_AREA|cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0,0,0)
                )

                pts = np.array([self.corners],np.int32)
                cv2.polylines(current, [pts], True, (0,255,0), 5)

                current = self.resize_to_smaller(current, self.monitor.width-self.screenPadding, self.monitor.height-self.screenPadding)

                cv2.imshow('current',current)        

                diff_image = self.resize_to_smaller(diff_image, self.monitor.width-self.screenPadding, self.monitor.height-self.screenPadding)

                cv2.imshow('diff image',diff_image)



                cv2.imshow('projector',projected)


                self.progressbar.stop()
                self.progressbar['mode'] = "determinate"
                self.progressbar.step(99.9)
            
                k = cv2.waitKey(1)

            except Exception as e:
                print(traceback.format_exc())

    def resize_to_larger(self, image, width, height):

        target_size = max(width, height)
        h, w = image.shape[:2]
        
        if h > w:
            scale = target_size / h
        else:
            scale = target_size / w
            
        new_width = int(w * scale)
        new_height = int(h * scale)
    
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    def resize_to_smaller(self, image, width, height):
       target_size = min(width, height)
       h, w = image.shape[:2]
       
       if h > w:
           scale = target_size / h
       else:
           scale = target_size / w
           
       new_width = int(w * scale)
       new_height = int(h * scale)

       return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    def crop_with_aspect_ratio(self,image, aspect_ratio):

        window_name = 'Select region (drag mouse) - Press ENTER when done, ESC to cancel'
        clone = image.copy()
        clone = self.resize_to_smaller(clone, self.monitor.width-self.screenPadding, self.monitor.height-self.screenPadding)

        cv2.namedWindow(window_name)
        
        # Initialize rectangle coordinates
        rect_coords = []
        drag_in_progress = False
        start_x, start_y = -1, -1
        
        def mouse_callback(event, x, y, flags, param):
            nonlocal drag_in_progress, start_x, start_y, rect_coords
            
            if event == cv2.EVENT_LBUTTONDOWN:
                drag_in_progress = True
                start_x, start_y = x, y
                
            elif event == cv2.EVENT_MOUSEMOVE and drag_in_progress:
                img_copy = clone.copy()
                
                # Calculate rectangle dimensions maintaining aspect ratio
                current_width = x - start_x
                current_height = int(abs(current_width) / aspect_ratio)
                
                if y < start_y:  # If dragging upward
                    current_height = -current_height
                    
                # Draw rectangle
                cv2.rectangle(img_copy, 
                             (start_x, start_y),
                             (start_x + current_width, start_y + current_height),
                             (0, 255, 0), 2)
                cv2.imshow(window_name, img_copy)
                
            elif event == cv2.EVENT_LBUTTONUP:
                drag_in_progress = False
                rect_coords = [
                    start_x,
                    start_y,
                    x,
                    start_y + int((x - start_x) / aspect_ratio)
                ]
        
        cv2.setMouseCallback(window_name, mouse_callback)
        
        while True:
            cv2.imshow(window_name, clone if not rect_coords else cv2.rectangle(
                clone.copy(),
                (rect_coords[0], rect_coords[1]),
                (rect_coords[2], rect_coords[3]),
                (0, 255, 0), 2
            ))
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # ENTER key
                break
            elif key == 27:  # ESC key
                rect_coords = []
                break
        
        cv2.destroyWindow(window_name)
        
        if not rect_coords:
            return image
            
        # Ensure coordinates are in the correct order (top-left to bottom-right)
        x1, y1, x2, y2 = rect_coords
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
            
        # Crop and return the image
        image = self.resize_to_smaller(clone, self.monitor.width-self.screenPadding, self.monitor.height-self.screenPadding)
        return image[y1:y2, x1:x2]



    def get_corrected_aspect_ratio(self, corners):
        dst = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        M = cv2.getPerspectiveTransform(corners, dst)
        
        test_points = np.float32([[0, 0], [1000, 0], [1000, 1000], [0, 1000]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(test_points, M)
        
        height = np.linalg.norm(transformed[1] - transformed[0])
        width = np.linalg.norm(transformed[3] - transformed[0])
        
        
        return width , height


    def get_roi_from_clicks(self):
        clicked_points = []
        
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                clicked_points.append((x, y))
                cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Click ROI', display_frame)
        
        ret, display_frame = self.cap.read()
        if not ret:
            return None
            
        cv2.imshow('Click ROI', display_frame)
        cv2.setMouseCallback('Click ROI', mouse_callback)
        
        while len(clicked_points) < 4:
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to cancel
                cv2.destroyWindow('Click ROI')
                return None
        
        cv2.destroyWindow('Click ROI')
        return clicked_points

    def get_subpixel_position(self,frame, approx_x, approx_y, window_size=10):
        # Extract region around approximate position
        y1 = max(0, approx_y - window_size)
        y2 = min(frame.shape[0], approx_y + window_size)
        x1 = max(0, approx_x - window_size)
        x2 = min(frame.shape[1], approx_x + window_size)
        roi = frame[y1:y2, x1:x2]
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Create coordinate meshgrid
        y, x = np.mgrid[0:roi.shape[0], 0:roi.shape[1]]
        
        # Calculate weighted centroid
        total = roi.sum()
        if total == 0:
            return approx_x, approx_y
            
        cx = np.sum(x * roi) / total + x1
        cy = np.sum(y * roi) / total + y1
        
        return cx, cy

    def registerProjector(self,withCrop=True):
        self.projectorSurface = np.zeros((self.projector.height,self.projector.width,3),np.uint8)
        
        if self.projector == self.monitor:
            cv2.namedWindow('projector')
        else:
            cv2.namedWindow('projector', cv2.WINDOW_NORMAL)

        cv2.imshow("projector", self.projectorSurface) 
        cv2.moveWindow("projector", self.projector.x+10, self.projector.y+10) 
        cv2.waitKey(1) 
        
        if self.projector != self.monitor:
            cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN, 1)
        
        cv2.waitKey(1) 
        cv2.setMouseCallback('projector', self.select_point, self.points)

        try: 
            if self.regdata is None:
                regdata = json.load(open('registrationData.json','r'))        
                res=mb.askquestion('Load last image registrations?', 'Load the camera, projector and canvas positions from the last run?')
                if res == 'yes' :
                    self.regdata = regdata
                    self.points = regdata['points']
                    self.corners = regdata['corners']
        except:
            pass


        if len(self.points) >= 4:
            self.projectorSurface[:] = (20, 20, 20)
            self.projectorSurface[:] = (0, 0, 0)
            cv2.fillPoly(self.projectorSurface, np.int32([self.points]),(1, 255, 1))
            cv2.imshow("projector", self.projectorSurface) 
            k = cv2.waitKey(100)
        
        k = cv2.waitKey(500)

        while len(self.points) < 4:
            self.projectorSurface[:] = (20, 20, 20)
            for i, point in enumerate(self.points):
                cv2.circle(self.projectorSurface, point, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(self.projectorSurface, self.points[i-1], point, (0, 255, 0), 1)
                cv2.line(self.projectorSurface, point, [self.last_mx,self.last_my], (0, 255, 0), 1)

                print(self.last_mx,self.last_my)

            cv2.imshow("projector", self.projectorSurface) 
            k = cv2.waitKey(1)

        self.projectorSurface[:] = (0, 0, 0)

        for i, point in enumerate(self.points):
            cv2.circle(self.projectorSurface, point, 4, (0, 255, 0), -1)

        cv2.imshow("projector", self.projectorSurface) 
        k = cv2.waitKey(100)

        self.target_points = np.float32(self.points)
        self.rect = np.zeros((4, 2), dtype=np.float32)

        s = self.target_points.sum(axis=1)
        self.rect[0] = self.points[np.argmin(s)]
        self.rect[2] = self.points[np.argmax(s)]

        diff = np.diff(self.target_points, axis=1)
        self.rect[1] = self.points[np.argmin(diff)]
        self.rect[3] = self.points[np.argmax(diff)]


        k = cv2.waitKey(100)
        ret, frame = self.cap.read()
        k = cv2.waitKey(1)

        spoints = np.array(self.target_points)
        min_x, max_x = np.min(spoints[:,0]), np.max(spoints[:,0])
        min_y, max_y = np.min(spoints[:,1]), np.max(spoints[:,1])

        
        if len(self.corners)<4:
            self.corners = self.get_roi_from_clicks()    
            for i,((cx,cy),(px,py)) in enumerate(zip(self.corners,self.points)):
                self.projectorSurface[:] = (0, 0, 0)
                self.projectorSurface[py-2:py+2,px-2:px+2] = (255, 255, 255)
                cv2.imshow("projector", self.projectorSurface) 
                cv2.waitKey(500)
                ret, frame = self.cap.read()
                refined_x, refined_y = self.get_subpixel_position(frame, cx, cy)
                self.corners[i] =  refined_x, refined_y
        

        json.dump({'points':self.points,'corners':self.corners},open('registrationData.json','w'))

        print(f"{self.points=}")
        print(f"{self.corners=}")

        self.projectorSurface[:] = (0, 0, 0)
        cv2.imshow("projector", self.projectorSurface) 

        target_points2 = np.float32(self.corners)
        self.rect2 = np.zeros((4, 2), dtype=np.float32)

        s = target_points2.sum(axis=1)
        self.rect2[0] = self.corners[np.argmin(s)]  # Top left (smallest sum)
        diff = np.diff(target_points2, axis=1)
        self.rect2[1] = self.corners[np.argmin(diff)] # Top right
        self.rect2[2] = self.corners[np.argmax(s)]    # Bottom right (largest sum)
        self.rect2[3] = self.corners[np.argmax(diff)] # Bottom left



        self.keystone_w, self.keystone_h = self.get_corrected_aspect_ratio(self.rect)
        self.canvas_w, self.canvas_h = self.get_corrected_aspect_ratio(self.rect2)


        self.ave_w =  (self.keystone_w+self.canvas_w)/2
        self.ave_h =  (self.keystone_h+self.canvas_h)/2

        self.ave_ratio = self.ave_w/self.ave_h

        if withCrop:
            cropped_target = self.crop_with_aspect_ratio(self.target,self.ave_ratio)
        else:
            cropped_target = self.target

        self.target_w = cropped_target.shape[1]
        self.target_h = cropped_target.shape[0]


        self.target_ratio = self.target_w / self.target_h

        print(f'ave_ratio',self.ave_ratio)
        print(f'target_ratio',self.target_ratio)

        self.zoom = 1

        if self.ave_w/self.ave_h < self.target_w/self.target_h:
            # Height limited - zoom width
            new_w = int(self.target_h * (self.ave_w/self.ave_h) / self.zoom)
            center = self.target_w // 2
            half_width = new_w // 2
            self.target_img = cropped_target[:, center-half_width:center+half_width]
        else:
            # Width limited - zoom height
            new_h = int(self.target_w / (self.ave_w/self.ave_h) / self.zoom)
            center = self.target_h // 2
            half_height = new_h // 2
            self.target_img = cropped_target[center-half_height:center+half_height, :]


        self.target_img = self.resize_to_larger(self.target_img,int(max_x - min_x),int(max_y - min_y))


        self.surface_w,self.surface_h = self.projectorSurface.shape[:2]
        self.target_w,self.target_h = self.target_img.shape[:2]



        self.matrix = cv2.getPerspectiveTransform(self.rect2, 
                                             np.float32([
                                                [0,0],
                                                [self.target_h,0],  
                                                [self.target_h,self.target_w],  
                                                [0,self.target_w]
                                            ]))
        


        print('cameraMatrix')

        current = self.getframe()

        self.source_points = np.float32([
            [0,0],
            [self.target_h,0],  
            [self.target_h,self.target_w],  
            [0,self.target_w]   
        ])            

        self.projectorMatrix = cv2.getPerspectiveTransform(self.rect,self.source_points)
        

        self.projected = cv2.warpPerspective(
            self.target_img,
            self.projectorMatrix,
            (self.surface_h,self.surface_w),
            flags=cv2.INTER_AREA|cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0)
        )


        self.projected_grey = cv2.warpPerspective(
            np.full_like(self.target_img,0),
            self.projectorMatrix,
            (self.surface_h,self.surface_w),
            flags=cv2.INTER_AREA|cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0,0,0)
        )
        self.recalc=True
        if self.thread is None:
            self.thread = threading.Thread(target=self._capture_loop)
            self.thread.daemon = True
            self.thread.start()

if __name__ == "__main__":
    root = tk.Tk()
    root.title('Painter\'s Diff')
    app = Controller(root)
    root.mainloop()

