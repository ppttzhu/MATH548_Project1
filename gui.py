from tkinter import *
from tkinter import messagebox

def click():
    message='Strike Price: '+strike.get()+'\nMaturity Date: '+maturity.get()\
    +'Call/Put Type: '+str(call_put_type.get())+'\nExercise Type: '+str(exercise_type.get())\
    +'Market Price: '+market_price.get()
    messagebox.showinfo('Price', message)#place holder

font=('TIMES NEW ROMAN',32)

window = Tk() 
window.title('Options Pricing Tool')

lbl = Label(window, text='Strike Price',font=font)
lbl.grid(column=0, row=0)
strike = Entry(window,width=10,font=font)
strike.grid(column=1, row=0)

lbl = Label(window, text='Maturity Date',font=font)
lbl.grid(column=0, row=1)
maturity = Entry(window,width=10,font=font)
maturity.grid(column=1, row=1)

call_put_type=IntVar()
rad1 = Radiobutton(window,text='Call', value=1, font=font, variable=call_put_type)
rad2 = Radiobutton(window,text='Put', value=2, font=font, variable=call_put_type)
rad1.grid(column=0, row=2) 
rad2.grid(column=1, row=2)
 

exercise_type=IntVar()
radn1 = Radiobutton(window,text='European', value=1, font=font, variable=exercise_type)
radn2 = Radiobutton(window,text='American', value=2, font=font, variable=exercise_type)
radn1.grid(column=0, row=3) 
radn2.grid(column=1, row=3)

lbl = Label(window, text='Market Price',font=font)
lbl.grid(column=0, row=4)
market_price = Entry(window,width=10,font=font)
market_price.grid(column=1, row=4)

btn = Button(window, text='Get Price', font=font, command=click)
btn.grid(column=0, row=5)

window.geometry('500x350')
window.mainloop()