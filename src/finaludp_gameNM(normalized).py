import socket
import struct
import tkinter as tk
import math


# Works with udp_sender.cpp

# UDP configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 12345

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

# Window and canvas size
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

# Create the main window
root = tk.Tk()
root.title("Joint Angle Target Plot")

# Create a canvas widget to draw the targets and the cursor
canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
canvas.pack()

# Define target radius and position
target_radius = 120
cursor_radius = 40


def angle_to_canvas_coords(sabd_deg, ef_deg):
    """Map SABD (0 to 60°) and EF (0° to 120°) to canvas X, Y coordinates."""
    x = int((sabd_deg) /12 * WINDOW_WIDTH) #this changed window width t0 -20 t0 +20
    y = int((1-ef_deg / 4) * WINDOW_HEIGHT) #caps window height at 4
    return x, y

# Define target angles
targets_def = [
    {"sabd": 0, "ef": 4},   # sets coordinates for targets
    {"sabd": 12, "ef": 0}     # sets coordinates for targets
]

# Draw targets
targets = {}
for t in targets_def:
    x, y = angle_to_canvas_coords(t["sabd"], t["ef"])
    tid = canvas.create_rectangle(
        x - target_radius, y - target_radius,
        x + target_radius, y + target_radius,
        fill="red"
    )
    targets[tid] = (x, y)

canvas.create_text(60, 60, text="EF", font=("Arial Bold", 20), fill="black")
canvas.create_text(540, 545, text="SABD", font=("Arial Bold", 20), fill="black")

# Create the simulated cursor (a small circle)
cursor_x, cursor_y = angle_to_canvas_coords(-20 , 0)
cursor = canvas.create_oval(
    cursor_x - cursor_radius, cursor_y - cursor_radius,
    cursor_x + cursor_radius, cursor_y + cursor_radius,
    fill="blue"
)

# Function to check if the simulated cursor is within a target's bounds
def is_within_target(cx, cy, tx, ty):
    return ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 <= target_radius

# Function to move the cursor and update the target colors
def move_cursor(new_x, new_y):
    global cursor_x, cursor_y
    cursor_x = max(0, min(WINDOW_WIDTH, new_x))
    cursor_y = max(0, min(WINDOW_HEIGHT, new_y))
    canvas.coords(cursor, cursor_x - cursor_radius, cursor_y - cursor_radius,
                  cursor_x + cursor_radius, cursor_y + cursor_radius)

    # Update target color if cursor is within target
    for tid, (tx, ty) in targets.items():
        hit = is_within_target(cursor_x, cursor_y, tx, ty)
        canvas.itemconfig(tid, fill="green" if hit else "red")

def receive_udp_data():
    try:
        data, addr = sock.recvfrom(1024)
        values = struct.unpack('28d', data)

        # Values in radians from sender
        shoulder_abduction_rad = values[5]  # index 5
        elbow_flexion_rad = values[(5 * 2)+1]       # index 11

        # Convert to degrees
        sabd_deg = shoulder_abduction_rad #to get correct values, they had to be negative
        ef_deg = elbow_flexion_rad #to get correct values, they had to be negative

        # Clamp values
        #sabd_deg = max(-60, min(60, sabd_deg)) #caps cursor between -60 and 60 for x (max had to be - and min had to be + for sum reason idk)
        #ef_deg = max(0, min(120, ef_deg)) #same here, max had to be zero and min had to be 120

        # Convert to canvas coordinates
        x, y = angle_to_canvas_coords(sabd_deg, ef_deg)
        move_cursor(x, y)

    except Exception as e:
        print(f"Error receiving joint angle data: {e}")

    # Schedule the next UDP check
    root.after(1, receive_udp_data)

# Start receiving UDP data
root.after(50, receive_udp_data)

# Start the main event loop
root.mainloop()
