import random
todolist=[]

 
def show_list():
    print("to add a task press 1")
    print("to remove a task press 2")
    print("to mark task as done press 3")
    print("to view the existing tasks press 4")
    print("to exit press 5")
show_list()



while True :
    choice=input("enter the option number you want ")
    if choice=="1":
        task=input("add the new task :")
        todolist.append({"task":task,"done":False})
        print("task added sucessfully")
    
    
    
    elif choice=="2":
        if len(todolist)==0:
            print("the list is already empty")
        else: print("choose the number of the task you want to delete ")
        for i, t in enumerate(todolist, start=1):
            print(f"{i}.{t['task']} ,{t['done']}") 
            try:
                task_num = int(input("Enter the task number to remove: "))
                removed = todolist.pop(task_num - 1)
                print("task removed")
            except (ValueError, IndexError):
                print(" Invalid task number.")
                
                
                
                
                
    elif choice=="3":
        if len(todolist)==0:
            print("no tasks to show")
        else:
            print("your to do list is ")
            for i,t in enumerate(todolist,start=1):
                print(f"{i}. {t['task']} ,{t['done']}")
                
                
                
                
                

                    
     
       