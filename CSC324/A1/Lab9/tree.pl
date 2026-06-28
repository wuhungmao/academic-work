inorder(nil, []).
inorder(tree(Left, Root, Right), List) :-
    inorder(Left, LList),
    inorder(Right, RList),
    append(LList, [Root|RList], List).

preorder(nil, []).
preorder(tree(Left, Root, Right), List) :-
    preorder(Left, LList),
    preorder(Right, RList),
    append([Root|LList], RList, List).

postorder(nil, []).
postorder(tree(Left, Root, Right), List) :-
    postorder(Left, LList),
    postorder(Right, RList),
    append(LList, RList, tmp),
    append(tmp, [Root], List).      

leaves(nil, []).
leaves(tree(nil, Root, nil), [Root]).
leaves(tree(Left, Root, Right), List) :-
    leaves(Left, Lleaves),
    leaves(Right, Rleaves),
    append(Lleaves, Rleaves, List).

contains(_, nil) :- fail.
contains(Val, tree(nil, Val, nil)).
contains(Val, tree(Left, Root, Right)) :-
    Val = Root ;
    contains(Val, Left) ;
    contains(Val, Right).


subtree(nil, _).
subtree(tree(nil, Val, nil), tree(nil, Val, nil)).
subtree(Subtree, tree(Leftrt, Rootrt, Rightrt)) :-
    same_tree(Subtree, tree(Leftrt, Rootrt, Rightrt));
    same_tree(Subtree, Leftrt);
    same_tree(Subtree, Rightrt).

same_tree(nil, nil).
same_tree(tree(Leftl, V, Rightl), tree(Leftr, V, Rightr)) :-
    same_tree(Leftl, Leftr),
    same_tree(Rightl, Rightr).
