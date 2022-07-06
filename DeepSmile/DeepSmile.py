from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import functions
import model_applay 
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfile, askopenfilename,asksaveasfilename
import tkinter
import tkinter.messagebox
import customtkinter
from fpdf import FPDF

import sklearn.utils._typedefs


customtkinter.set_appearance_mode("Light")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("dark-blue")  # Themes: "blue" (standard), "green", "dark-blue"
class App(customtkinter.CTk):

    WIDTH = 1200
    HEIGHT = 700
    

    def __init__(self):
        super().__init__()

        self.title("DeepSmile")
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}+{0}+{0}")
        self.iconbitmap("logo1.ico")
        self.menubar = tkinter.Menu(self)
        self.helpmenu = tkinter.Menu(self.menubar, tearoff=0)
        # self.logo_1 = ImageTk.PhotoImage(Image.open('feder1.png'))
        # self.logo_1  = customtkinter.CTkLabel(self.feder_w , image=self.logo_1)
        # img = ImageTk.PhotoImage(file="feder1.png") 
       # self.helpmenu.add_command(label="Financing",image=self.logo_1)#command=lambda:self.feder())
        self.helpmenu.add_command(label="Financing",command=lambda:self.feder())
        self.menubar.add_cascade(label="About", menu=self.helpmenu)
        self.config(menu=self.menubar)
        

        
        self.dataset_test = None
        self.rmse_test_mean = None
        self.rmse_mm = None
        self.sujet_param = None
        
 # ============ create two frames ============
        # configure grid layout (2x1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.frame_left = customtkinter.CTkFrame(master=self,width=380,)
        self.frame_left.grid(row=0, column=0, sticky="nswe")

        self.frame_right = customtkinter.CTkFrame(master=self)
        self.frame_right.grid(row=0, column=1, sticky="nswe",padx=5 )

# ============ frame_left ============
                # configure grid layout (1x11)
        self.frame_left.grid_rowconfigure(0, minsize=10)   # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(8, weight=1)  # empty row as spacing
        self.frame_left.grid_rowconfigure(10, minsize=20)    # empty row with minsize as spacing
        self.frame_left.grid_rowconfigure(15, minsize=10)  # empty row with minsize as spacing
        
        self.logo = ImageTk.PhotoImage(Image.open('logo.png').resize((350, 150), ))
        self.logo_label = customtkinter.CTkLabel(self.frame_left,image=self.logo, )
        self.logo_label.grid(column=0, row=0,sticky="swe")

        self.name_label = customtkinter.CTkLabel(self.frame_left, text="Name", text_font=("Arial", -18)) .grid(row=2, column=0,pady=20,  sticky="W")
        self.name_entry = customtkinter.CTkEntry(self.frame_left,width=200, placeholder_text="Subject's Name")
        self.name_entry.grid(row=2, column=0,  pady=20, padx=20, sticky="E")

        self.sex_label = customtkinter.CTkLabel(self.frame_left, text="Sex", text_font=("Arial", -18)) .grid(row=3, column=0, pady=20,  sticky="W")
        self.sex_entry = customtkinter.CTkEntry(self.frame_left,width=200, placeholder_text="Male/Female")
        self.sex_entry.grid(row=3, column=0,  pady=20, padx=20, sticky="E")

        self.age_label = customtkinter.CTkLabel(self.frame_left, text="Date of \n Birth", text_font=("Arial", -18)) .grid(row=4, column=0, pady=20,  sticky="W")
        self.age_entry = customtkinter.CTkEntry(self.frame_left,width=200, placeholder_text="dd/mm/yy")
        self.age_entry.grid(row=4, column=0,  pady=20, padx=20, sticky="E")

        self.entre_btn = customtkinter.CTkButton(self.frame_left ,width=100, height=20, border_width=3,command=lambda: self.entre_click(),text="Enter",text_font=("Arial", 12),fg_color=("gray75", "gray30"))
        self.entre_btn.grid(column=0,row = 5, sticky="e",padx=20,)

        self.button_1 = customtkinter.CTkButton(self.frame_left, text="Select CSV file",command=lambda:self.open_file(),text_font=("Arial", 15), fg_color=("gray75", "gray30"),width=150, height=60, border_width=3)
        self.button_1.grid(row=8, column=0, pady=10, padx=20)
        
# ============ frame_right ============

        # configure grid layout (3x7)
        self.frame_right.rowconfigure(1, weight=1)
        self.frame_right.rowconfigure(1, weight=1)
        self.frame_right.columnconfigure((0, 1), weight=1)
        self.frame_right.columnconfigure(1, weight=0)
        
        self.frame_info = customtkinter.CTkFrame(self.frame_right)
        self.frame_info.grid(row=0, column=0, columnspan=2, pady=10, padx=10, sticky="nsew")
        
# ============ frame_info ============
        # configure grid layout (1x1)
        self.frame_info.rowconfigure(4, weight=1)
        self.frame_info.columnconfigure(3, weight=1)      
        self.label_info_1 = customtkinter.CTkLabel(master=self.frame_info, text="Result",text_font=("Arial", 15), height=15, fg_color=("white", "gray38"),
                                                           justify=tkinter.LEFT)
        self.label_info_1.grid(column=1, row=0,columnspan=1, sticky="S", padx=5, pady=5)
        
        self.text_box = tkinter.Text(self.frame_info, height=5, width=5, )
        self.text_box.config(state='disabled', font=("Colibry", 11))  
        self.text_box.tag_configure("center", justify="center")
        self.text_box.tag_add("center", 1.0, "end")
        self.text_box.grid(row=1, column=1,columnspan=5, sticky="NSWE", padx=15, pady=5)
        
        
        self.load_label = customtkinter.CTkLabel(self.frame_info, text="LOADING DATA...", text_font=("Arial", -18)) 
        #self.load_label.grid(row=2, column=1,padx=15, pady=5,  sticky="nsew")
        self.plot_button = customtkinter.CTkButton(self.frame_info,width=100, height=20, border_width=3,command= lambda: self.plot(),
                                              text="Show graph",text_font=("Arial", 12),fg_color=("gray75", "gray30"))
        self.plot_button .grid(row=3, column=3, pady=5, padx=5, sticky="E")

        self.open_button = customtkinter.CTkButton(self.frame_info,width=100, height=20, border_width=3, command=lambda:self.result(),
                                              text="Open result",text_font=("Arial", 12),fg_color=("gray75", "gray30"),)
        self.open_button .grid(row=3, column=4,  pady=5, padx=5, sticky="E")

        self.save_button = customtkinter.CTkButton(self.frame_info, height=20,border_width=3,text_font=("Arial", 12),fg_color=("gray75", "gray30"),
                                                       text="Save", command=lambda:self.save_all())
        self.save_button.grid(row=3, column=5,  pady=5, padx=5, sticky="E")
        
        # self.radio_var = tkinter.IntVar(0)
        # self.radio_var.set(0)
        # self.radio_button_1 = customtkinter.CTkRadioButton(self.frame_info, variable=self.radio_var,text="Markers plot", command=lambda:self.plot(),
        #                                                    text_font=("Arial", 10),value=0)
        # self.radio_button_1.grid(row=3, column=1,  pady=5, padx=5, sticky="n")

        # self.radio_button_2 = customtkinter.CTkRadioButton(self.frame_info,text="Radar plot",variable=self.radio_var,
        #                                                     value=1, text_font=("Arial", 10),  command=lambda:self.radar_plot())
        # self.radio_button_2.grid(row=3, column=2, pady=5, padx=5, sticky="n")
        
        self.frame_plot = customtkinter.CTkFrame(self.frame_right,fg_color=("white", "gray30"),width=800)
        self.frame_plot.grid(row=1, column=0, padx=10,sticky="SN")

        # self.frame_plot.grid_rowconfigure(8, weight=1)  # empty row as spacing
        self.frame_plot.rowconfigure(0, weight=0)
        self.frame_plot.columnconfigure(3, weight=1)

    def open_file(self):  
        self.load_label.grid(row=2, column=1,padx=15, pady=5,  sticky="nsew")
        file = askopenfile(parent=self.frame_left, mode='rb', title="Ouvrir", filetypes=[("Csv file", "*.csv")])
        # filesize = os.path.getsize(file)
        self.load_label.grid_remove()
        
        if file:       
            data_ref= functions.ref_frame(file)
            dP0_data = functions.ref_to_dP0(data_ref)
            dP0_l = []
            dP0_l.append(dP0_data)        
            dP0_list = functions.list_to_array(dP0_l)
            inter_list = functions.list_to_interpolate(dP0_list)
            scaled_list, max_value_list, min_value_list = functions.scaled_data2(inter_list)
            self.dataset_test, self.rmse_test_mean, self.rmse_mm = model_applay.model_app(scaled_list, max_value_list, min_value_list)     
         
            seuile =  0.05527*100
            if self.rmse_test_mean >= seuile:
                self.detected = "Anomaly detected"
            else:
                self.detected = "Anomaly not detected" 

            
            self.text_box = tkinter.Text(self.frame_info, height=5, width=5, )
            self.text =(f' Name: {self.sujet_param[0]}    Sex: {self.sujet_param[1]}    Age: {self.sujet_param[2]} \n' 
                        f' Degree of abnormality: {self.rmse_test_mean} % \n Degree of abnormality in millimeters: {self.rmse_mm} mm \n {self.detected} ')
            self.text_box.insert(1.0, self.text)
            self.text_box.config(state='disabled', font=("Arial", 15))  
            self.text_box.tag_configure("left", justify="center")
            self.text_box.tag_add("center", 1.0, "end")
            self.text_box.grid(row=1, column=1,columnspan=5, sticky="NSWE", padx=15, pady=5)
            
            
    def entre_click(self): 
         self.name = self.name_entry.get()
         self.sex = self.sex_entry.get()
         self.age = self.age_entry.get()
    
         self.name_entry.delete(0, 'end')
         self.sex_entry.delete(0, 'end')
         self.age_entry.delete(0, 'end')  
         self.sujet_param = [self.name,self.sex,self.age]
    
    def result(self):
        
        self.new_window_1 = customtkinter.CTkToplevel(self.frame_left)
        self.new_window_1.title("Result") 
        self.new_window_1.geometry("600x700")
        self.new_window_1.iconbitmap("logo1.ico")
        
        self.visual_res,self.graph = model_applay.multi_step_plot(self.dataset_test,self.rmse_test_mean,self.rmse_mm)
        self.visual_res.subplots_adjust( bottom=0.04, top=0.97)
        self.canvas = FigureCanvasTkAgg(self.visual_res, master = self.new_window_1)
         
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=1, column=1,sticky="NWSE")
        
        self.toolbarFrame = tkinter.Frame(self.new_window_1)
        self.toolbarFrame.grid(row=0, column=1,sticky="N")
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.toolbarFrame)

    def plot(self):
         self.visual,self.plt = model_applay.multi_step_plot(self.dataset_test,self.rmse_test_mean,self.rmse_mm)    
         self.visual.subplots_adjust( bottom=0.07,top=0.97)
         self.canvas1 = FigureCanvasTkAgg(self.visual, master = self.frame_plot)        
         self.canvas1.draw()
         self.canvas1.get_tk_widget().grid(row=0, rowspan=1, column=1,columnspan=3,)
         
    def feder(self):
            
        self.feder_w = customtkinter.CTkToplevel(self)
        self.feder_w.title("Finansing") 
        self.feder_w.geometry("550x150")
        self.feder_w.resizable(False, False)
        self.feder_w.iconbitmap("logo1.ico")
        self.logo_1 = ImageTk.PhotoImage(Image.open('feder1.png'))
        self.label_1  = customtkinter.CTkLabel(self.feder_w , image=self.logo_1)
        self.label_1.grid(column=1, row=0,sticky="nswe")
        
             
    def save_all(self):
        file_name = asksaveasfilename(filetypes=[("txt file", "*.txt")])
        f = open(file_name, 'w')

        d = self.text
        s =self.plt
        self.plt.savefig(file_name)
        f.write(d)
        f.close()
        
    # def save_all(self):
    #     pdf = FPDF()
    #     pdf.add_page()
    #     pdf.set_font("Arial", "", 16)
    #     pdf.cell(40, 10, self.text.strip())
    #     pdf.output(asksaveasfilename(filetypes=[("PDF file", "*.pdf")]), "F")
             
            
    def start(self):
        self.mainloop()
        #self.config()
            
if __name__ == "__main__":
    app = App()
    app.start()