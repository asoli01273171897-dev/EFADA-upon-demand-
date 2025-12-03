import os
import subprocess
import tkinter as tk 
from tkinter import PhotoImage
from tkinter import scrolledtext
from tkinter import ttk
import util
import cv2
from PIL import Image, ImageTk
import datetime

class App:
    

    def __init__(self):
        self.main_window = tk.Tk()
        self.main_window.geometry("1200x520+350+100")
        #self.main_window.configure(bg='purple')

                # Load the background image
        bg_image = Image.open(r"wallpaper/luke-chesser-eICUFSeirc0-unsplash.jpg")
        self.bg_photo = ImageTk.PhotoImage(bg_image)

        # Create a label to hold the background image
        bg_label = tk.Label(self.main_window, image=self.bg_photo)
        bg_label.place(relwidth=1, relheight=1)  # Cover the entire window








        self.signout_main_window = util.get_button(self.main_window, 'sign out', 'red', self.signout)
        self.signout_main_window.place(x=750, y=200)
        self.favicon_icon = util.get_icon_button(self.main_window, r"D:\downloads\face attendance\face attendance\icons\favicon.ico", self.generate_report, state=tk.DISABLED)
        self.favicon_icon.place(x=750, y=20)
        self.report_code_entry, self.report_code_var = util.get_password_entry2(self.main_window)
        self.report_code_entry.place(x=750, y=100)
        self.report_code_var.trace_add('write', self.check_password)







        

        # Use lowercase color names
        self.signin_button_main_window = util.get_button(self.main_window, 'sign in', 'gray', self.signin,)
        self.signin_button_main_window.place(x=750, y=300)

        self.signup_button_main_window = util.get_button(self.main_window, 'sign up', 'green', self.signup, fg='black')
        self.signup_button_main_window.place(x=750, y=400)

        self.web_cam_label = util.get_img_label(self.main_window)
        self.web_cam_label.place(x=10, y=0, width=700, height=500)
        self.add_webcam(self.web_cam_label)
        self.dp_dir = './dp'
        if not os.path.exists(self.dp_dir):
            os.makedirs(self.dp_dir)
        self.log_path = './log.txt'
        self.signouts_path = './signouts.txt'



    def check_password(self, *args):
        entered_password = self.report_code_var.get()

        # Check if the entered password matches the expected password
        if entered_password == "sss":
            # Enable the icon
            self.favicon_icon.config(state=tk.NORMAL)
        else:
            # Disable the icon
            self.favicon_icon.config(state=tk.DISABLED)



    def add_webcam(self, label):
        if 'cap' not in self.__dict__:
            self.cap = cv2.VideoCapture(0)

        self._label = label
        self.process_webcam()

    def generate_report(self):
        # Create a new window for the report
        report_window = tk.Toplevel(self.main_window)
        report_window.geometry("800x600")
        report_window.title("Attendance Report")

        # Create a Treeview widget to display the report in a table
        columns = ("Name", "Sign-in Time", "Sign-out Time", "Working period")
        report_tree = ttk.Treeview(report_window, columns=columns, show="headings")

        # Set column headings
        for col in columns:
            report_tree.heading(col, text=col)

        # Pack the Treeview
        report_tree.pack(fill=tk.BOTH, expand=1)

        # Display sign-ins and sign-outs with calculated time difference
        self.display_log_report(self.log_path, self.signouts_path, report_tree)

    def display_log_report(self, log_file_path, signouts_file_path, report_tree):
        try:
            with open(log_file_path, 'r') as log_file, open(signouts_file_path, 'r') as signouts_file:
                signins_lines = log_file.readlines()
                signouts_lines = signouts_file.readlines()

                for signin_line in signins_lines:
                    signin_name, signin_time_str = signin_line.strip().split(', ')
                    signin_time = datetime.datetime.strptime(signin_time_str, "%Y-%m-%d %H:%M:%S.%f")

                    # Find the corresponding sign-out time
                    signout_time = None
                    for signout_line in signouts_lines:
                        signout_name, signout_time_str = signout_line.strip().split(', ')
                        if signin_name == signout_name:
                            signout_time = datetime.datetime.strptime(signout_time_str, "%Y-%m-%d %H:%M:%S.%f")
                            break

                    if signout_time:
                        time_difference = signout_time - signin_time
                        row_data = (signin_name, signin_time_str, signout_time_str, str(time_difference))
                        report_tree.insert("", tk.END, values=row_data)
                    else:
                        row_data = (signin_name, signin_time_str, "No sign-out recorded", "not signed out yet")
                        report_tree.insert("", tk.END, values=row_data)

        except FileNotFoundError:
            report_tree.insert("", tk.END, values=("Log file(s) not found.", "", "", ""))

    def process_webcam(self):
        ret, frame = self.cap.read()
        self.most_recent_capture_arr = frame

        _img = cv2.cvtColor(self.most_recent_capture_arr, cv2.COLOR_BGR2RGB)
        self.most_recent_capture_PIL = Image.fromarray(_img)
        imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_PIL)
        self._label.imgtk = imgtk  # Keep a reference to prevent garbage collection
        self._label.configure(image=imgtk)

        # Schedule the next update after 20 milliseconds (adjust as needed)
        self._label.after(20, self.process_webcam)

    def signin(self):
        unknown_person_img = './.tmp.jpg'
        cv2.imwrite(unknown_person_img, self.most_recent_capture_arr)

        output = str(subprocess.check_output(['face_recognition', self.dp_dir, unknown_person_img]))
        elheta_el8sla=' \r\n'

        name = output.split(',')[1][:-5]



        if name  in ['unknown_person', 'no_pers']:
            util.msg_box('oops...!', 'unknown user please register as a new user or try again') 
        else:
            util.msg_box('welcome back', 'welcome,{}'.format(name))
            with open(self.log_path, 'a') as f:
                current_time = datetime.datetime.now()
                f.write('{}, {}\n'.format(name, current_time))
                f.close

        os.remove(unknown_person_img)

    def signup(self):
        self.signup_window = tk.Toplevel(self.main_window)
        self.signup_window.geometry("1200x520+370+120")

        self.accept_signup_button_signup_window = util.get_disabled_button(self.signup_window, 'confirm', 'green', self.confirm, fg='white')
        self.accept_signup_button_signup_window.place(x=750, y=400)

        self.try_again_signup_button_signup_window = util.get_button(self.signup_window, 'Try again', 'red', self.try_again, fg='black')
        self.try_again_signup_button_signup_window.place(x=750, y=300)

        self.cap_cam_label = util.get_img_label(self.signup_window)
        self.cap_cam_label.place(x=10, y=0, width=700, height=500)
        self.add_img_to_label(self.cap_cam_label)
        self.entry_text_signup = util.get_entry_text(self.signup_window)
        self.entry_text_signup.place(x=750, y=240)
        self.text_label_register_new_user1 = util.get_text_label(self.signup_window, 'please enter your name')
        self.text_label_register_new_user1.place(x=750, y=210)

        self.text_label_register_new_user2 = util.get_text_label(self.signup_window, 'please enter Admins code')
        self.text_label_register_new_user2.place(x=750, y=10)
        self.password_check_txt_field = util.get_password_entry(self.signup_window)
        self.password_check_txt_field.place(x=750, y=40)
        department_options = ["IT", "Customer Service", "Financial", "HR", "Trainee"]
        self.department_combobox = ttk.Combobox(self.signup_window, values=department_options)
        self.department_combobox.place(x=750, y=100)
        self.text_label_register_new_user3 = util.get_text_label(self.signup_window, 'please select your department')
        self.text_label_register_new_user3.place(x=750, y=70)

    def signout(self):
        unknown_person_img = './.tmp_signouts.jpg'  # Use a different filename for signout
        cv2.imwrite(unknown_person_img, self.most_recent_capture_arr)

        output = str(subprocess.check_output(['face_recognition', self.dp_dir, unknown_person_img]))

        name = output.split(',')[1][:-5]

        if name in ['unknown_person', 'no_persons_found']:
            util.msg_box('oops...!', 'unknown user, please sign in or try again')
        else:
            util.msg_box('goodbye', 'Goodbye, {}'.format(name))
            with open(self.signouts_path, 'a') as f:
                current_time = datetime.datetime.now()
                f.write('{}, {}\n'.format(name, current_time))
                f.close

        os.remove(unknown_person_img)

    def add_img_to_label(self, label):
         imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_PIL)
         label.imgtk = imgtk  # Keep a reference to prevent garbage collection
         label.configure(image=imgtk)
         self.signup_cap = self.most_recent_capture_arr.copy()

    def start(self):
        self.main_window.mainloop()

    def confirm(self):
        # Get the entered name and password
        name = self.entry_text_signup.get(1.0, "end-1c")
        entered_password = self.password_check_txt_field.get()

        # Check if the entered password is correct
        if entered_password == "sss":
            # Enable the signup button and proceed
            self.accept_signup_button_signup_window.config(state=tk.NORMAL)

            # Perform the signup actions
            cv2.imwrite(os.path.join(self.dp_dir, '{}.jpg'.format(name)), self.signup_cap)
            util.msg_box('welcome', 'Successfully signed up!')
            self.signup_window.destroy()
        else:
            # Password is incorrect, show a message or take appropriate action
            util.msg_box('Invalid Password', 'Please enter the correct password.')

    def try_again(self):
        self.signup_window.destroy()
    
if __name__ == "__main__":
    app = App()
    app.start()
