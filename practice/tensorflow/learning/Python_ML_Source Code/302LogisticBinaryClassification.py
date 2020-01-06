import numpy as np
from tkinter import *
import tkinter.scrolledtext as tkst
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import math
# Import pandas as a alias 'pd'
import pandas as pd

# Load the CSV files "marathon_results_2015 ~ 2017.csv" under "data" folder
marathon_2015_2017_qualifying = pd._________("./data/marathon_2015_2017_qualifying.csv")

# Merge 2015, 2016 and 2017 files into marathon_2015_2017 file index by Official Time
marathon_2015_2016 = marathon_2015_2017_qualifying[marathon_2015_2017_qualifying['Year'] __ 2017]
marathon_2017 = marathon_2015_2017_qualifying[marathon_2015_2017_qualifying['Year'] __ 2017]

df_2015_2016 = pd.DataFrame(marathon_2015_2016,______=['M/F',  'Age',  'Pace',  'qualifying'])
df_2017 = pd.DataFrame(marathon_2017,_______=['M/F',  'Age',  'Pace',  'qualifying'])

# Dataframe to List
record_2015_2016 = df_2015_2016.values.tolist()
record_2017 = df_2017.values.tolist()

gender_list = ['Female', 'Male']
grad_fig = Figure(figsize=(10, 6), dpi=100)
grad_ax = grad_fig.add_subplot(111)
grad_ax.set_xlim(15, 88)
grad_ax.set_ylim(0, 1300)
grad_ax.set_ylabel("Pace : Runner's overall minute per mile pace")
grad_ax.set_xlabel("Age : Age on race day")

def seconds_to_hhmmss(seconds):
    hours = seconds // (60*60)
    seconds %= (60*60)
    minutes = seconds // 60
    seconds %= 60
    return "%02i:%02i:%02i" % (hours, minutes, seconds)

def normalization(record):
    r0 = record[0]
    r1 = record[1] / 10
    r2 = record[2] / 100
    return [r0, r1, r2]

# X and Y data from 0km to 30km    
# x_train = [ r[0:3] for r in record_2015_2016]
x_train = [ normalization(r[0:3]) for r in record_2015_2016]
y_train = [ [r[-1]] for r in record_2015_2016]
# x_test = [ r[0:3] for r in record_2017]
x_test = [ r[0:3] for r in record_2017]
y_test = [ [r[-1]] for r in record_2017]

def histogram():
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a]
    print(runner)
    t_g = int(runner[0])
    t_y = int(runner[1])
    t_p = int(runner[2])
    if(t_g):
        gender_color = 'b'
    else:
        gender_color = 'r'  
    gender_record = df_2017[df_2017['M/F'] == ___]
    gender_age_record = gender_record[gender_record.Age == ___] 
    gender_age_record_list = gender_age_record.values.tolist() 
    
    grad_ax.plot(gender_record.Age, gender_record.Pace, '.', color=gender_color, alpha=0.5)
    grad_ax.plot(t_y, t_p, 'yd')
    stat = gender_age_record['Pace']._________
    print(stat)
    title = 'Gender : '+gender_list[t_g]+', Age : '+str(t_y)+', Pace : '+str(t_p)
    grad_ax.set_title(title)
    grad_ax.annotate('['+gender_list[t_g]+', '+str(t_y)+']', (75, 1200), fontsize=10)
    grad_ax.annotate("%10s %7i" % ('Count : ', stat[0]), (75, 1150), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('Mean :  ', stat[1]), (75, 1100), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('25% :   ', stat[3]), (75, 1050), fontsize=10)
    grad_ax.annotate("%10s %7.3f" % ('75% :   ', stat[5]), (75, 1000), fontsize=10)
        
    grad_fig.canvas.draw()    
    
def learing(): 
    """
    MAchine Learning, Tensorflow 
    """
    # Tensorflow Linear Regression
    import tensorflow as tf
    tf.set_random_seed(777)  # for reproducibility
    
    t_a = int(t_aSpbox.get()) - 1
    runner = x_test[t_a]
    t_g = int(runner[0])
    t_y = int(runner[1])
    t_p = int(runner[2])

    t_t = int(t_tSpbox.get()) + 1
    t_r = float(t_rSpbox.get())

    # placeholders for a tensor 
    W = tf.Variable(tf.random_normal([____]), name='weight')
    b = tf.Variable(tf.random_normal([_]), name='bias')
    
    X = tf.placeholder(tf.float32, shape=[None, _])
    Y = tf.placeholder(tf.float32, shape=[None, _])

    # Hypothesis
    hypothesis = tf.______(tf.matmul(X, W) + b)

    # Simplified cost/loss function
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                           tf.log(1 - hypothesis))
    
    # optimizer
    train = tf.train.____________________(learning_rate=___).minimize(____)
    
    # Accuracy computation
    # True if hypothesis>0.5 else False
    predicted = tf.cast(_________ > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(________, Y), dtype=tf.float32))

    # Launch a session.
    with tf.Session() as sess:
        # Initializes global variables 
        sess.run(tf.______________________)
    
        # Fit the line
        log_ScrolledText.insert(END, '\nGender :'+gender_list[t_g]+', Age :'+str(t_y)+', Pace :'+str(t_p)+'\n', 'TITLE')
        log_ScrolledText.insert(END, '\n\nCost Decent & Accuracy\n\n','HEADER')
        log_ScrolledText.insert(END, "%10s %20s %20s" % ('Step', 'Cost', 'Accuracy(%)')+'\n\n')
        for step in range(t_t):
            _, cost_val, h_val, p_val, a_val = sess.run([train, cost, _________, _________, ________], feed_dict={X: x_train, Y: y_train})
            
            if step % 100 == 0:
                print(step, cost_val, a_val) 
                log_ScrolledText.insert(END, "%10i %20.5f %20.7f" % (step, cost_val, a_val*100)+'\n')
        
        winner = [ t_g, t_y, t_p ]
        result = sess.run(hypothesis, feed_dict={X: [normalization(winner)]})
        log_ScrolledText.insert(END, '\n\n')
        log_ScrolledText.insert(END, "%10s %20s" % ('Value        ', 'Qualifying Prediction\n\n'), 'HEADER')
        if(result[0] > 0.5):
            log_ScrolledText.insert(END, "%10.7f %20s" % (result[0], 'Qualifier\n\n'), 'Qualifier')
        else:
            log_ScrolledText.insert(END, "%10.7f %20s" % (result[0], 'DisQualifier\n\n'), 'DisQualifier')        
    
#main
main = Tk()
main.title("Logistic Regression Binary Classification")
main.geometry()

label=Label(main, text='Logistic Regression Binary Classification')
label.config(font=("Courier", 18))
label.grid(row=0,column=0,columnspan=6)

t_aVal  = IntVar(value=1)
t_aSpbox = Spinbox(main, textvariable=t_aVal ,from_=0, to=len(x_test), increment=1, justify=RIGHT)
#t_aSpbox.config(state='readonly')
t_aSpbox.grid(row=1,column=1)
t_aLabel=Label(main, text='Rank of runner : ')                
t_aLabel.grid(row=1,column=0)

t_tVal  = IntVar(value=10000)
t_tSpbox = Spinbox(main, textvariable=t_tVal ,from_=0, to=100000, increment=1000, justify=RIGHT)
#t_tSpbox.config(state='readonly')
t_tSpbox.grid(row=1,column=3)
t_tLabel=Label(main, text='Number of train : ')                
t_tLabel.grid(row=1,column=2)

t_rVal  = DoubleVar(value=1e-2)
t_rSpbox = Spinbox(main, textvariable=t_rVal ,from_=0, to=1, increment=1e-2, justify=RIGHT)
#t_rSpbox.config(state='readonly')
t_rSpbox.grid(row=1,column=5)
t_rLabel=Label(main, text='Learning rate : ')                
t_rLabel.grid(row=1,column=4)

Button(main,text="Histogram", height=2,command=lambda:histogram()).grid(row=2, column=0, columnspan=3, sticky=(W, E))
Button(main,text="Prediction", height=2,command=lambda:learing()).grid(row=2, column=3, columnspan=3, sticky=(W, E))

grad_canvas = FigureCanvasTkAgg(grad_fig, main)
grad_canvas.get_tk_widget().grid(row=3,column=0,columnspan=6)

log_ScrolledText = tkst.ScrolledText(main, height=15)
log_ScrolledText.grid(row=4,column=0,columnspan=6, sticky=(N, S, W, E))
log_ScrolledText.configure(font='TkFixedFont')
log_ScrolledText.tag_config('Qualifier', foreground='blue', font=("Helvetica", 16))
log_ScrolledText.tag_config('DisQualifier', foreground='red', font=("Helvetica", 16))
log_ScrolledText.tag_config('HEADER', foreground='gray', font=("Helvetica", 14), underline=1)
log_ScrolledText.tag_config('TITLE', foreground='orange', font=("Helvetica", 18), underline=1, justify='center')

main.mainloop()