# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1rwRodZwPF_nvzFWWG7nGqlcsMCjAk9fX
"""

List=["neelu","nidhi","amitht"]
print(List)

from typing import Concatenate
List1=[28,"nidhi",6,78,"arr"]
List2=[4,5,6,7,8]
Concatenate=List1 + List2
print(Concatenate)

t=['r','y','u']
t[:1]

marks=[45,67,89]
age=[12,17,18]
hours=[4,8,10]

my_list=[1,'hello',2.23]
print("Original List:",my_list)
my_list.append('world')
print("List after appending 'world':", my_list)
my_list.remove(2.23)
print("List after removing 3.14:",my_list)

my_tuple=(91,'apple',3.14)
print("Original Tuple:",my_tuple)
print("First element:",my_tuple[0])

for i in range(1,6):
  print(i)

i=1
while i<=5:
  print(i)
  i+=1

num=int(input("Enter a number: "))
if num>=0:
  print("positive number")
else:
  print("negative number")

def greet(name):
  print(f"Hello from function, {name}!")
greet("nidhi")