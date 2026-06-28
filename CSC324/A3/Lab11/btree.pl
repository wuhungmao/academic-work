count_nodes(Node, 1).
count_nodes(t(Left, Node, Right), N) :-
    count_nodes(Left, NLeft),
    count_nodes(Right, NRight),
    N is NLeft + NRight + 1.

sum_keys(nil, 0).
sum_keys(t(Left, Node, Right), N) :-
    sum_keys(Left, NLeft),
    sum_keys(Right, NRight),
    N is NLeft + NRight + Node.

max_val(Left, Right, Left) :- 
    Left >= Right.

max_val(Left, Right, Right) :- 
    Left < Right.

height(nil, 0).
height(t(Left, Node, Right), N) :-
    height(Left, NLeft),
    height(Right, NRight),
    max_val(NLeft, NRight, Result),
    N is Result + 1.

contains(t(_, Match, _), Match).
contains(t(Left, Node, _), N) :-
    N < Node,
    contains(Left, N).
contains(t(_, Node, Right), N) :-
    N > Node,
    contains(Right, N).

not_contains(Tree, K) :-
    \+ contains(Tree, K).

