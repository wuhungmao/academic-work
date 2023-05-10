from __future__ import annotations

from typing import Optional, Any


class BinarySearchTree:
    """Binary Search Tree class.
    This class represents a binary tree satisfying the Binary Search Tree
    property: for every item, its value is >= all items stored in its left
    subtree, and <= all items stored in its right subtree.
    """
    # === Private Attributes ===
    # The item stored at the root of the tree, or None if the tree is empty.
    _root: Optional[Any]
    # The left subtree, or None if the tree is empty.
    _left: Optional[BinarySearchTree]
    # The right subtree, or None if the tree is empty.
    _right: Optional[BinarySearchTree]

    # === Representation Invariants ===
    # - If self._root is None, then so are self._left and self._right.
    # This represents an empty BST.
    # - If self._root is not None, then self._left and self._right
    # are BinarySearchTrees.
    # - (BST Property) If self is not empty, then
    # all items in self._left are <= self._root, and
    # all items in self._right are >= self._root.
    def __init__(self, root=None, left=None, right=None):
        self._root = root
        self._left = left
        self._right = right

    def closest(self, item: int) -> Optional[int]:
        """Return the value in this BST that is closest to <item>.
        By "closest" here, we mean the value that minimizes the absolute distance to <item>.
        If there is a tie, return the smaller value.
        Return None if this BST is empty.
        Precondition: this BST contains only integers.
        >>> bst = BinarySearchTree(10, BinarySearchTree(3, BinarySearchTree(2), BinarySearchTree(7)), BinarySearchTree(32, BinarySearchTree(27), BinarySearchTree(81, BinarySearchTree(49), BinarySearchTree(99))))
        >>> items = 20
        >>> bst.closest(items)
        27
        """
        if self._root is None:
            return None
        if self._left is None and item <= self._root:
            return self._root
        if self._right is None and item >= self._root:
            return self._root
        if self._left is None and self._right is None:
            return self._root
        else:
            if item <= self._root:
                return self._left.closest(item)
            else:
                return self._right.closest(item)


def count_odd(obj) -> int:
    """Return the number of odd numbers in <obj>.
    >>> count_odd([1, [2, 6, 5], [9, [8, 7]]])
    4
    """
    if isinstance(obj, int):
        if obj % 2 == 0:
            return 0
        else:
            return 1
    else:
        sum = 0
        for sublist in obj:
            sum += count_odd(sublist)
        return sum




if __name__ == '__main__':
    import doctest

    doctest.testmod()
