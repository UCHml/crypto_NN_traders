import customtkinter
from PIL import Image

customtkinter.set_appearance_mode('dark')
customtkinter.set_default_color_theme('green')

app = customtkinter.CTk()

app.geometry('900x650')
app.resizable(False, False)

app.title('Crypton')


def start_trading():
    # Includes loading data, training the model, making predictions and portfolio management
    # We use functions and logic from other modules

    # Display a message about the successful start of trading
    # messagebox.showinfo("Trading App", "Automatic trading has started!")
    pass


logo = customtkinter.CTkImage(dark_image=Image.open("snk2b.png"), size=(150, 150))
logo_label = customtkinter.CTkLabel(master=app, text='', image=logo)
logo_label.grid(row=0, column=0, padx=380, pady=60)

btn = customtkinter.CTkButton(master=app,
                              text="Start Trading",
                              width=190,
                              height=110,
                              border_width=1,
                              border_color='black',
                              text_color='black',
                              font=('BinancePlex,Arial,sans-serif', 20, 'bold'),
                              command=start_trading)
btn.grid(row=1, column=0, padx=10, pady=10)

app.mainloop()
