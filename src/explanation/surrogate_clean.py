f = """
digraph Tree {
node [shape=box, style="rounded", color="black", fontname=helvetica] ;
edge [fontname=helvetica] ;
0 [label="Number of leadership experiences <= 1.5"] ;
1 [label="Essay score <= 5.25"] ;
0 -> 1 [labeldistance=2.5, labelangle=45, headlabel="True"] ;
3 [label="GPA <= 3.46"] ;
1 -> 3 [headlabel="True     "] ;
7 [label="Average score:
 3.42"] ;
3 -> 7 [headlabel="False     "] ;
8 [label=""] ;
3 -> 8 [headlabel="True     "] ;
4 [label="Grade is not College/University Grad Student"] ;
1 -> 4 [headlabel="False     "] ;
13 [label="Average score:
 5.06"] ;
4 -> 13 ;
14 [label="Average score:
 6.49"] ;
4 -> 14 ;
2 [label="Essay score <= 4.75value = 6.18"] ;
0 -> 2 [labeldistance=2.5, labelangle=-45, headlabel="False"] ;
5 [label="GPA <= 3.55"] ;
2 -> 5 [headlabel="True     "] ;
11 [label="Average score:
 4.62"] ;
5 -> 11 ;
12 [label="value = 5.67"] ;
5 -> 12 ;
6 [label="GPA <= 3.65"] ;
2 -> 6 [headlabel="False     "] ;
9 [label="Average score:
 5.59"] ;
6 -> 9 [headlabel="True     "] ;
10 [label="Average score:
 6.87"] ;
6 -> 10 ;
}"""


import re

number = 2


def remove_value_from_node(f, node):

    m = re.search(f'{node} \[label=".*?value = \d+\.\d+"\]', f)
    string = m.group()

    m_subset = re.search(f'value = \d+\.\d+', string)
    subset = m_subset.group()

    string_new = string.replace(subset, '')
    f = f.replace(string, string_new)
    print(string, string_new)

    return f




for node in [2, 12]:
    f = remove_value_from_node(f, node)



