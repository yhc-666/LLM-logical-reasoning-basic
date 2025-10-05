from z3 import *

objects_sort, (Anne, Bob, Dave, Gary) = EnumSort('objects', ['Anne', 'Bob', 'Dave', 'Gary'])
attributes_sort, (big, red, rough, smart, blue, furry, young) = EnumSort('attributes', ['big', 'red', 'rough', 'smart', 'blue', 'furry', 'young'])
objects = [Anne, Bob, Dave, Gary]
attributes = [big, red, rough, smart, blue, furry, young]
has_attribute = Function('has_attribute', objects_sort, attributes_sort, BoolSort())

pre_conditions = []
pre_conditions.append(has_attribute(Anne, big) == True)
pre_conditions.append(has_attribute(Anne, red) == True)
pre_conditions.append(has_attribute(Anne, rough) == True)
pre_conditions.append(has_attribute(Anne, smart) == False)
pre_conditions.append(has_attribute(Bob, big) == True)
pre_conditions.append(has_attribute(Dave, big) == True)
pre_conditions.append(has_attribute(Dave, blue) == True)
pre_conditions.append(has_attribute(Dave, furry) == True)
pre_conditions.append(has_attribute(Dave, red) == True)
pre_conditions.append(has_attribute(Dave, rough) == False)
pre_conditions.append(has_attribute(Dave, smart) == False)
pre_conditions.append(has_attribute(Dave, young) == True)
pre_conditions.append(has_attribute(Gary, big) == True)
pre_conditions.append(has_attribute(Gary, blue) == True)
pre_conditions.append(has_attribute(Gary, furry) == True)
pre_conditions.append(has_attribute(Gary, young) == True)
x = Const('x', objects_sort)
pre_conditions.append(ForAll([x], Implies(has_attribute(x, young) == True, has_attribute(x, furry) == True)))
x = Const('x', objects_sort)
pre_conditions.append(ForAll([x], Implies(And(has_attribute(x, red) == True, has_attribute(x, big) == True), has_attribute(x, furry) == True)))
x = Const('x', objects_sort)
pre_conditions.append(ForAll([x], Implies(has_attribute(x, furry) == True, has_attribute(x, red) == True)))
x = Const('x', objects_sort)
pre_conditions.append(ForAll([x], Implies(And(has_attribute(x, young) == True, has_attribute(x, blue) == False), has_attribute(x, red) == False)))
pre_conditions.append(Implies(has_attribute(Bob, big) == True, has_attribute(Bob, young) == True))
pre_conditions.append(Implies(And(has_attribute(Bob, red) == True, has_attribute(Bob, furry) == False), has_attribute(Bob, rough) == False))
x = Const('x', objects_sort)
pre_conditions.append(ForAll([x], Implies(And(has_attribute(x, red) == True, has_attribute(x, young) == False), has_attribute(x, rough) == True)))
x = Const('x', objects_sort)
pre_conditions.append(ForAll([x], Implies(And(has_attribute(x, furry) == True, has_attribute(x, smart) == True), has_attribute(x, rough) == False)))

def is_valid(option_constraints):
    solver = Solver()
    solver.add(pre_conditions)
    solver.add(Not(option_constraints))
    return solver.check() == unsat

def is_unsat(option_constraints):
    solver = Solver()
    solver.add(pre_conditions)
    solver.add(option_constraints)
    return solver.check() == unsat

def is_sat(option_constraints):
    solver = Solver()
    solver.add(pre_conditions)
    solver.add(option_constraints)
    return solver.check() == sat

def is_accurate_list(option_constraints):
    return is_valid(Or(option_constraints)) and all([is_sat(c) for c in option_constraints])

def is_exception(x):
    return not x


if is_valid(has_attribute(Anne, furry) == False): print('(A)')
if is_unsat(has_attribute(Anne, furry) == False): print('(B)')