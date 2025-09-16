import socket
import struct
import tkinter as tk
import time
import errno

# UDP configuration
UDP_IP = "0.0.0.0"
UDP_PORT = 12345

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)  # This is critical

# Window and canvas size
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600

# Create the main window
root = tk.Tk()
root.title("Joint Angle Target Plot")

canvas = tk.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
canvas.pack()

# Target and cursor settings
target_radius = 120
base_cursor_radius = 40
min_z_torque = -5.0
max_z_torque = 5.0

# Duration the cursor must remain in target (in seconds)
REQUIRED_HOLD_DURATION = 1.0

# Game state control
game_paused = False
pause_timer = None
hold_start_time = None
active_target_id = None
receive_after_id = None

# Continue button (initially hidden)
continue_button = tk.Button(root, text="Continue", font=("Arial", 16), command=lambda: resume_game())
continue_button.place_forget()

def angle_to_canvas_coords(sabd_deg, ef_deg):
    x = int((sabd_deg) / 1 * WINDOW_WIDTH)
    y = int((1 - ef_deg / 4) * WINDOW_HEIGHT)
    return x, y

# Define 2D target positions
targets_def = [
    {"sabd": 0, "ef": 4},
    {"sabd": 1, "ef": 0}
]

targets = {}
for t in targets_def:
    x, y = angle_to_canvas_coords(t["sabd"], t["ef"])
    tid = canvas.create_rectangle(
        x - target_radius, y - target_radius,
        x + target_radius, y + target_radius,
        fill="red"
    )
    targets[tid] = (x, y)

# Z target (static visual cue)
z_target_x, z_target_y = angle_to_canvas_coords(0, 4)
z_target_radius = base_cursor_radius * 0.5
canvas.create_oval(
    z_target_x - z_target_radius, z_target_y - z_target_radius,
    z_target_x + z_target_radius, z_target_y + z_target_radius,
    outline="black", width=2
)

# Labels
canvas.create_text(60, 60, text="EF", font=("Arial Bold", 20), fill="black")
canvas.create_text(540, 545, text="SABD", font=("Arial Bold", 20), fill="black")
canvas.create_text(z_target_x, z_target_y - 30, text="SF Target (Z max)", font=("Arial", 12), fill="black")

# Cursor
cursor_x, cursor_y = angle_to_canvas_coords(0, 0)
cursor_radius = base_cursor_radius
cursor = canvas.create_oval(
    cursor_x - cursor_radius, cursor_y - cursor_radius,
    cursor_x + cursor_radius, cursor_y + cursor_radius,
    fill="blue"
)

def is_within_target(cx, cy, tx, ty):
    return ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 <= target_radius

def pause_game():
    global game_paused, receive_after_id
    game_paused = True
    if receive_after_id:
        root.after_cancel(receive_after_id)
    continue_button.place(relx=0.5, rely=0.9, anchor="center")

def resume_game():
    global game_paused, hold_start_time, active_target_id
    game_paused = False
    hold_start_time = None
    active_target_id = None
    continue_button.place_forget()
    schedule_receive()

def move_cursor(new_x, new_y, sf_torque):
    global hold_start_time, active_target_id

    new_x = max(0, min(WINDOW_WIDTH, new_x))
    new_y = max(0, min(WINDOW_HEIGHT, new_y))

    # Scale cursor size
    norm_sf = (sf_torque - min_z_torque) / (max_z_torque - min_z_torque)
    norm_sf = max(0.0, min(1.0, norm_sf))
    scaled_radius = base_cursor_radius * (1.5 - norm_sf)

    canvas.coords(cursor,
                  new_x - scaled_radius, new_y - scaled_radius,
                  new_x + scaled_radius, new_y + scaled_radius)

    now = time.time()
    in_any_target = False

    for tid, (tx, ty) in targets.items():
        hit = is_within_target(new_x, new_y, tx, ty)
        canvas.itemconfig(tid, fill="green" if hit else "red")

        if hit:
            in_any_target = True
            if active_target_id == tid:
                if hold_start_time and now - hold_start_time >= REQUIRED_HOLD_DURATION:
                    pause_game()
                    return
            else:
                active_target_id = tid
                hold_start_time = now
        else:
            if active_target_id == tid:
                active_target_id = None
                hold_start_time = None

def receive_udp_data():
    global receive_after_id
    if game_paused:
        return

    try:
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                values = struct.unpack('28d', data)

                sabd_rad = values[5]
                ef_rad = values[11]
                sf_torque = values[9]

                sabd_deg = sabd_rad
                ef_deg = ef_rad

                x, y = angle_to_canvas_coords(sabd_deg, ef_deg)
                move_cursor(x, y, sf_torque)

            except BlockingIOError:
                break  # No more data to read

            except socket.error as e:
                if e.errno == errno.EWOULDBLOCK:
                    break
                else:
                    print(f"Socket error: {e}")
                    break

    except Exception as e:
        print(f"General error receiving joint angle data: {e}")

    schedule_receive()

def schedule_receive():
    global receive_after_id
    receive_after_id = root.after(20, receive_udp_data)

schedule_receive()
root.mainloop()
