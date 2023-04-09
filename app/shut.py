from tkinter import *
from PIL import ImageTk, Image
import cv2
from final import abc
root = Tk()
root.title('Shut Your Mouth')
root.iconbitmap('icon.ico')
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry('%dx%d+0+0' % (w, h))
root.configure(bg='black')
frame1 = LabelFrame(root, text='Camera')
frame1.place(x=50, y=50)
L1 = Label(frame1)
L1.pack(fill='both', expand='yes')
frame2 = LabelFrame(root, text='Lips Segmentation')
frame2.place(x=100, y=600)
L2 = Label(frame2)
L2.pack(fill='both', expand='yes')
frame3 = LabelFrame(root, text='Contours')
frame3.place(x=400, y=600)
L3 = Label(frame3)
L3.pack(fill='both', expand='yes')
frame4 = LabelFrame(root, text='aa')
frame4.place(x=800, y=600)
s = Scale(frame4, from_=4000, to=12000, tickinterval=500, orient=HORIZONTAL, length=500)
s.set(8000)
s.pack()
cap = cv2.VideoCapture(0)
c = 0
logic= abc()
while True:
    image = cap.read()[1]
    image, lips, contours, c = logic.getresponse(image, int(s.get()), int(c))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = ImageTk.PhotoImage(Image.fromarray(image))
    lips = cv2.cvtColor(lips, cv2.COLOR_BGR2RGB)
    lips = ImageTk.PhotoImage(Image.fromarray(lips))
    contours = cv2.cvtColor(contours, cv2.COLOR_BGR2RGB)
    contours = ImageTk.PhotoImage(Image.fromarray(contours))
    L1['image'] = image
    L2['image'] = lips
    L3['image'] = contours
    root.update()

cap.release()