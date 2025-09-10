from z3 import *

people_sort, (Anne, Bob, Gary, Harry) = EnumSort('people', ['Anne', 'Bob', 'Gary', 'Harry'])
attributes_sort, (quiet, rough, big, kind, young, furry, round) = EnumSort('attributes', ['quiet', 'rough', 'big', 'kind', 'young', 'furry', 'round'])
people = [Anne, Bob, Gary, Harry]
attributes = [quiet, rough, big, kind, young, furry, round]
has_attribute = Function('has_attribute', people_sort, attributes_sort, BoolSort())

pre_conditions = []
pre_conditions.append(has_attribute(Anne, quiet) == True)
pre_conditions.append(has_attribute(Bob, rough) == False)
pre_conditions.append(has_attribute(Gary, big) == True)
pre_conditions.append(has_attribute(Gary, kind) == True)
pre_conditions.append(has_attribute(Gary, rough) == True)
pre_conditions.append(has_attribute(Gary, young) == True)
pre_conditions.append(has_attribute(Harry, young) == True)
x = Const('x', people_sort)
pre_conditions.append(ForAll([x], Implies(has_attribute(x, big) == True, has_attribute(x, furry) == True)))
x = Const('x', people_sort)
pre_conditions.append(ForAll([x], Implies(has_attribute(x, young) == True, has_attribute(x, furry) == True)))
x = Const('x', people_sort)
pre_conditions.append(ForAll([x], Implies(And(has_attribute(x, quiet) == True, has_attribute(x, kind) == True), has_attribute(x, furry) == True)))
pre_conditions.append(Implies(And(has_attribute(Harry, furry) == True, has_attribute(Harry, quiet) == True), has_attribute(Harry, round) == True))
x = Const('x', people_sort)
pre_conditions.append(ForAll([x], Implies(And(has_attribute(x, rough) == True, has_attribute(x, kind) == True), has_attribute(x, quiet) == True)))
x = Const('x', people_sort)
pre_conditions.append(ForAll([x], Implies(And(has_attribute(x, young) == True, has_attribute(x, rough) == True), has_attribute(x, kind) == True)))
x = Const('x', people_sort)
pre_conditions.append(ForAll([x], Implies(And(has_attribute(x, quiet) == True, has_attribute(x, furry) == True), has_attribute(x, round) == True)))
x = Const('x', people_sort)
pre_conditions.append(ForAll([x], Implies(has_attribute(x, furry) == True, has_attribute(x, rough) == True)))

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


if is_valid(has_attribute(Harry, quiet) == False): print('(A)')
if is_unsat(has_attribute(Harry, quiet) == False): print('(B)')