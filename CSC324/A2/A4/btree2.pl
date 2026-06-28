% =============================================
% CSC324 Assignment: Binary Search Trees in Prolog
% File: btree2.pl
% =============================================

is_bst(nil).
is_bst(t(Left_subtr, Root_node, Right_subtr)) :-
    all_keys_less(Left_subtr, Root_node),
    all_keys_greater(Right_subtr, Root_node),
    is_bst(Left_subtr),
    is_bst(Right_subtr).

all_keys_less(nil, _). 
all_keys_less(t(Left_subtr, Subtr_root_node, Right_subtr), Root_node) :-
    Subtr_root_node < Root_node,                  
    all_keys_less(Left_subtr, Root_node),  
    all_keys_less(Right_subtr, Root_node).

all_keys_greater(nil, _). 
all_keys_greater(t(Left_subtr, Subtr_root_node, Right_subtr), Root_node) :-
    Subtr_root_node > Root_node,                   
    all_keys_greater(Left_subtr, Root_node),  
    all_keys_greater(Right_subtr, Root_node). 



strictly_balanced(nil).
strictly_balanced(t(Left_subtr, _, Right_subtr)) :-
    % Find left subtree and right subtree height
    height(Left_subtr, Left_subtr_height),
    height(Right_subtr, Right_subtr_height),
    % Find the difference
    Diff_height is Left_subtr_height - Right_subtr_height,
    % Check if difference is acceptable
    (Diff_height = -1; Diff_height = 0; Diff_height = 1),
    strictly_balanced(Left_subtr),
    strictly_balanced(Right_subtr).

% Height helper function: took from lab 11
max_val(Left, Right, Left) :- 
    Left >= Right.

max_val(Left, Right, Right) :- 
    Left < Right.

height(nil, 0).
height(t(Left, _, Right), Height) :-
    height(Left, NLeft),
    height(Right, NRight),
    max_val(NLeft, NRight, Max),
    Height is Max + 1.

% Find the Kth smallest key in a BST
kth_smallest(Tree, K, Value) :-
    % in order traversal gives us ascending list
    inorder(Tree, List),
    % can't use builtin nth1, so have to define it ourself
    element_at(K, List, Value).

element_at(1, [Head|_], Head).
element_at(K, [_|Tail], Element) :-
    K > 1,
    Next_k is K - 1,
    element_at(Next_k, Tail, Element).

% took from lab 10
inorder(nil, []).
inorder(t(Left, Root, Right), List) :-
    inorder(Left, LList),
    inorder(Right, RList),
    append(LList, [Root|RList], List).

% only match when left subtree or right subtree is null, which we return a small tree with k as root
insert_bst(nil, K, t(nil, K, nil)).
insert_bst(t(Left_subtr, Key, Right_subtr), K, t(New_left_subtr, Key, Right_subtr)) :-
    K < Key,                    
    insert_bst(Left_subtr, K, New_left_subtr).  

insert_bst(t(Left_subtr, Key, Right_subtr), K, t(Left_subtr, Key, New_right_subtr)) :-
    K > Key,                    
    insert_bst(Right_subtr, K, New_right_subtr).

% Like A2, and A3, I work with Saabit Zubairi, his utorid is zubairis