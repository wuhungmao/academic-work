"""
Assignment 2 starter code
CSC148, Winter 2022
Instructors: Bogdan Simion, Sonya Allin, and Pooja Vashisth

This code is provided solely for the personal and private use of
students taking the CSC148 course at the University of Toronto.
Copying for purposes other than this use is expressly prohibited.
All forms of distribution of this code, whether as given or with
any changes, are expressly prohibited.

All of the files in this directory and all subdirectories are:
Copyright (c) 2022 Bogdan Simion, Dan Zingaro
"""
from __future__ import annotations

import time

from huffman import HuffmanTree
from utils import *

import cProfile


# ====================
# Functions for compression


def build_frequency_dict(text: bytes) -> dict[int, int]:
    """ Return a dictionary which maps each of the bytes in <text> to its
    frequency.

    >>> d = build_frequency_dict(bytes([99]))
    >>> d == {99:1}
    True
    """
    byte_dict = {}
    for byte in text:
        if byte not in byte_dict:
            byte_dict[byte] = 1
        elif byte in byte_dict:
            byte_dict[byte] += 1
    return byte_dict


def build_huffman_tree(freq_dict: dict[int, int]) -> HuffmanTree:
    """ Return the Huffman tree corresponding to the frequency dictionary
    <freq_dict>.

    Precondition: freq_dict is not empty.

    >>> freq = {76: 76, 29: 29, 49: 49}
    >>> t = build_huffman_tree(freq)
    >>> result = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> t
    HuffmanTree(None, HuffmanTree(76, None, None), HuffmanTree(None,
    HuffmanTree(29, None, None), HuffmanTree(49, None, None)))
    >>> freq = {2: 1, 3: 1, 7:2, 4:1, 5:1,}
    >>> t = build_huffman_tree(freq)
    >>> t
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(6, None, None),
    HuffmanTree(7, None, None)), HuffmanTree(None,
    HuffmanTree(None, HuffmanTree(2, None, None), HuffmanTree(3, None, None)),
    HuffmanTree(None, HuffmanTree(4, None, None), HuffmanTree(5, None, None))))
    >>> import random
    >>> symbol = random.randint(0,255)
    >>> freq = {symbol: 6}
    >>> t = build_huffman_tree(freq)
    >>> any_valid_byte_other_than_symbol = (symbol + 1) % 256
    >>> dummy_tree = HuffmanTree(any_valid_byte_other_than_symbol)
    >>> result = HuffmanTree(None, HuffmanTree(symbol), dummy_tree)
    >>> t.left == result.left or t.right == result.left
    True
    """
    freq_list = list(freq_dict.items())
    main_huffman_tree = HuffmanTree()
    if len(freq_dict) == 1:
        solely_huffman_tree = HuffmanTree(freq_list[0][0])
        if freq_list[0][0] < 255:
            dummy_huffman_tree = HuffmanTree(symbol=freq_list[0][0] + 1)
        else:
            dummy_huffman_tree = HuffmanTree(symbol=freq_list[0][0] - 1)
        main_huffman_tree = HuffmanTree()
        main_huffman_tree.left = dummy_huffman_tree
        main_huffman_tree.right = solely_huffman_tree
        return main_huffman_tree
    else:
        while len(freq_list) != 1:
            min_item = min(freq_list, key=lambda x: x[1])
            smaller_item = freq_list.pop(freq_list.index(min_item))
            min_item = min(freq_list, key=lambda x: x[1])
            bigger_item = freq_list.pop(freq_list.index(min_item))
            if not isinstance(smaller_item[0], HuffmanTree):
                small_huffman_tree = HuffmanTree(smaller_item[0])
                small_huffman_tree_freq = freq_dict[smaller_item[0]]
            else:
                small_huffman_tree = smaller_item[0]
                small_huffman_tree_freq = smaller_item[1]
            if not isinstance(bigger_item[0], HuffmanTree):
                big_huffman_tree = HuffmanTree(bigger_item[0])
                big_huffman_tree_freq = freq_dict[bigger_item[0]]
            else:
                big_huffman_tree = bigger_item[0]
                big_huffman_tree_freq = bigger_item[1]
            main_huffman_tree = HuffmanTree(symbol=None,
                                            left=small_huffman_tree,
                                            right=big_huffman_tree)
            freq_list.append((main_huffman_tree,
                              small_huffman_tree_freq + big_huffman_tree_freq))
        return main_huffman_tree


def _get_codes_version_2(tree: HuffmanTree, level: list[str],
                         code_dict: dict) -> dict[int, str]:
    """version 2 of get code
    >>> left = HuffmanTree(None, HuffmanTree(9), HuffmanTree(3))
    >>> right = HuffmanTree(2)
    >>> tree = HuffmanTree(None, left, right)
    >>> d = _get_codes_version_2(tree, [''], {})
    >>> d == {9: "00", 3:'01', 2:'1'}
    True
    """
    if tree.right is None and tree.left is None:
        code_dict[tree.symbol] = level[0][:]
        return code_dict
    else:
        if isinstance(tree.left, HuffmanTree):
            level[0] = level[0] + '0'
            code_dict.update(_get_codes_version_2(tree.left, level, code_dict))
            level[0] = level[0][0:-1]
        if isinstance(tree.right, HuffmanTree):
            level[0] = level[0] + '1'
            code_dict.update(_get_codes_version_2(tree.right, level, code_dict))
            level[0] = level[0][0:-1]
    return code_dict


def get_codes(tree: HuffmanTree) -> dict[int, str]:
    """ Return a dictionary which maps symbols from the Huffman tree <tree>
    to codes.

    >>> tree = HuffmanTree(None, HuffmanTree(None, HuffmanTree(6, None, None), HuffmanTree(None, HuffmanTree(7, None, None), HuffmanTree(8, None, None))), HuffmanTree(9, None, None))
    >>> d = get_codes(tree)
    >>> d
    {5: '0', 6: '1'}
    """
    if tree.left is None and tree.right is None:
        return {tree.symbol: '0'}
    level = ['']
    code_dict = {}
    return _get_codes_version_2(tree, level, code_dict)




def _number_nodes_version_2(tree: HuffmanTree, the_num: int) -> int:
    """Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.
    >>> left = HuffmanTree(None, HuffmanTree(None,HuffmanTree(3),
    HuffmanTree(5)), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> the_tree = HuffmanTree(None, left, right)
    >>> _number_nodes_version_2(the_tree, 0)
    4
    >>> the_tree.left.left.number
    0
    >>> the_tree.left.number
    1
    >>> the_tree.right.number
    2
    >>> the_tree.number
    3
    """
    if tree.left.symbol is not None and tree.right.symbol is not None:
        tree.number = the_num
        return the_num + 1
    else:
        if tree.left.symbol is None:
            the_num = _number_nodes_version_2(tree.left, the_num)
        if tree.right.symbol is None:
            the_num = _number_nodes_version_2(tree.right, the_num)
        tree.number = the_num
        return the_num + 1


def number_nodes(tree: HuffmanTree) -> None:
    """ Number internal nodes in <tree> according to postorder traversal. The
    numbering starts at 0.

    >>> left = HuffmanTree(None, HuffmanTree(3), HuffmanTree(2))
    >>> right = HuffmanTree(None, HuffmanTree(9), HuffmanTree(10))
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> tree.left.number
    0
    >>> tree.right.number
    1
    >>> tree.number
    2
    """
    the_num = 0
    _number_nodes_version_2(tree, the_num)


def avg_length(tree: HuffmanTree, freq_dict: dict[int, int]) -> float:
    """ Return the average number of bits required per symbol, to compress the
    text made of the symbols and frequencies in <freq_dict>, using the Huffman
    tree <tree>.

    The average number of bits = the weighted sum of the length of each symbol
    (where the weights are given by the symbol's frequencies), divided by the
    total of all symbol frequencies.

    >>> freq = {3: 2, 2: 3, 9: 2, 4:5, 7:10, 11:4}
    >>> tree2 = build_huffman_tree(freq)
    >>> tree2
    HuffmanTree(None, HuffmanTree(7, None, None), HuffmanTree(None,
    HuffmanTree(None, HuffmanTree(2, None, None), HuffmanTree(11, None, None)),
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(3, None, None),
    HuffmanTree(9, None, None)), HuffmanTree(4, None, None))))
    >>> avg_length(tree2, freq)
    2.3846153846153846
    """
    code_dict = get_codes(tree)
    freq_dict_list = freq_dict.items()
    the_sum = 0
    for symbols in freq_dict_list:
        the_sum += freq_dict[symbols[0]] * len(code_dict[symbols[0]])
    denominator = sum(freq_dict.values())
    return the_sum / denominator


def _get_bits(text: bytes, codes: dict[int, str]) -> list[str]:
    """
    >>> d = {0: "0", 1: "11", 2: "100", 3:'101'}
    >>> text = bytes([1, 2, 1, 3])
    >>> bits_lst = _get_bits(text, d)
    >>> bits_lst == ['11100111','01000000']
    True
    >>> the_bytes = bytes([bits_to_byte(bits) for bits in bits_lst])
    >>> text = bytes([3,3,2,0,3,1])
    >>> result = _get_bits(text, d)
    >>> result == ['10110110','00101110']
    True
    """
    the_string = ''
    for num in text:
        the_string = the_string + codes[num]
    the_list = []
    if len(the_string) < 8:
        bits = the_string + '0' * (8 - len(the_string))
        return [bits]
    for index in range(0, len(the_string), 8):
        if index + 8 > len(the_string):
            the_list.append(the_string[index:])
        else:
            the_list.append(the_string[index:index + 8])
    if len(the_list[-1]) < 8:
        the_list[-1] = the_list[-1] + '0' * (8 - len(the_list[-1]))
    return the_list


def compress_bytes(text: bytes, codes: dict[int, str]) -> bytes:
    """ Return the compressed form of <text>, using the mapping from <codes>
    for each symbol.

    >>> d = {0: "0", 1: "11", 2: "100", 3:'101'}
    >>> text = bytes([1, 2, 1, 3])
    >>> bits_lst = _get_bits(text, d)
    >>> bits_lst == ['11100111','01000000']
    True
    >>> lst_bytes = bytes([bits_to_byte(bits) for bits in bits_lst])
    >>> compress_bytes(text, d) == lst_bytes
    True
    >>> test2 = bytes([231, 64])
    >>> lst_bytes == test2
    True
    >>> text = bytes([3,3,2,0,3,1])
    >>> result = _get_bits(text, d)
    >>> result == ['10110110','00101110']
    True
    >>> lst_bytes = bytes([bits_to_byte(bits) for bits in result])
    >>> test3 = bytes([182, 46])
    >>> compress_bytes(text, d) == test3
    True
    """
    bits_list = _get_bits(text, codes)
    sum_lst = []
    for bits in bits_list:
        sum_lst.append(bits_to_byte(bits))
    return bytes(sum_lst)


def _add_to_bytes_lst(tree: HuffmanTree, bytes_lst: list[int],
                      num_looking_for: int) -> list:
    if tree.left.is_leaf():
        bytes_lst.append(0)
        bytes_lst.append(tree.left.symbol)
    else:
        bytes_lst.append(1)
        bytes_lst.append(tree.left.number)
    if tree.right.is_leaf():
        bytes_lst.append(0)
        bytes_lst.append(tree.right.symbol)
    else:
        bytes_lst.append(1)
        bytes_lst.append(tree.right.number)
    return [bytes_lst, num_looking_for + 1]


def _tree_to_bytes_version_2(tree: HuffmanTree, bytes_lst: list[int],
                             num_looking_for: int) -> list[list[int], int]:
    if tree.number == num_looking_for:
        return _add_to_bytes_lst(tree, bytes_lst, num_looking_for)
    else:
        if not tree.left.is_leaf():
            bytes_and_num = _tree_to_bytes_version_2(tree.left, bytes_lst,
                                                     num_looking_for)
            bytes_lst = bytes_and_num[0]
            num_looking_for = bytes_and_num[1]
        if not tree.right.is_leaf():
            bytes_and_num = _tree_to_bytes_version_2(tree.right, bytes_lst,
                                                     num_looking_for)
            bytes_lst = bytes_and_num[0]
            num_looking_for = bytes_and_num[1]
        if tree.number == num_looking_for:
            return _add_to_bytes_lst(tree, bytes_lst, num_looking_for)
    return []


def tree_to_bytes(tree: HuffmanTree) -> bytes:
    """ Return a bytes representation of the Huffman tree <tree>.
    The representation should be based on the postorder traversal of the tree's
    internal nodes, starting from 0.

    Precondition: <tree> has its nodes numbered.

    >>> tree = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2]
    >>> left = HuffmanTree(None, HuffmanTree(3, None, None), \
    HuffmanTree(2, None, None))
    >>> right = HuffmanTree(5)
    >>> tree = HuffmanTree(None, left, right)
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 3, 0, 2, 1, 0, 0, 5]
    >>> tree = build_huffman_tree(build_frequency_dict(b"helloworld"))
    >>> number_nodes(tree)
    >>> list(tree_to_bytes(tree))
    [0, 104, 0, 101, 0, 119, 0, 114, 1, 0, 1, 1, 0, 100, 0, 111, 0, 108, 1, 3,
    1, 2, 1, 4]
    """
    bytes_list = []
    num_looking_for = 0
    bytes_list = _tree_to_bytes_version_2(tree, bytes_list, num_looking_for)[0]
    return bytes(bytes_list)


def compress_file(in_file: str, out_file: str) -> None:
    """ Compress contents of the file <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f1:
        text = f1.read()
    freq = build_frequency_dict(text)
    tree = build_huffman_tree(freq)
    codes = get_codes(tree)
    number_nodes(tree)
    print("Bits per symbol:", avg_length(tree, freq))
    result = (tree.num_nodes_to_bytes() + tree_to_bytes(tree)
              + int32_to_bytes(len(text)))
    result += compress_bytes(text, codes)
    with open(out_file, "wb") as f2:
        f2.write(result)


# ====================
# Functions for decompression

def generate_tree_general(node_lst: list[ReadNode],
                          root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes nothing about the order of the tree nodes in the list.

    >>> lst = [ReadNode(1, 3, 0, 7), ReadNode(0, 10, 0, 12),
    ReadNode(1, 1, 1, 0), ReadNode(0, 1, 0, 2)]
    >>> generate_tree_general(lst, 2)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(10, None, None),
    HuffmanTree(12, None, None)), HuffmanTree(None, HuffmanTree(None,
    HuffmanTree(1, None, None), HuffmanTree(2, None, None)),
    HuffmanTree(7, None, None)))
    """
    root = node_lst[root_index]
    main_huffman_tree = HuffmanTree()
    main_huffman_tree.number = root_index
    if root.l_type == 0:
        left_huffman_tree = HuffmanTree(root.l_data)
        main_huffman_tree.left = left_huffman_tree
    elif root.l_type == 1:
        main_huffman_tree.left = generate_tree_general(node_lst, root.l_data)
    if root.r_type == 0:
        right_huffman_tree = HuffmanTree(root.r_data)
        main_huffman_tree.right = right_huffman_tree
    elif root.r_type == 1:
        main_huffman_tree.right = generate_tree_general(node_lst, root.r_data)
    return main_huffman_tree


def _build_huffman_tree_from_readnode(node: ReadNode, index: int,
                                      main_huffman_trees: list):
    if node.l_type == 0:
        left_tree = HuffmanTree(node.l_data)
    elif node.l_type == 1 and node.r_type == 1:
        left_tree = main_huffman_trees.pop(-2)
    else:
        popped_node_left = main_huffman_trees.pop(-1)
        left_tree = popped_node_left
    if node.r_type == 0:
        right_tree = HuffmanTree(node.r_data)
    else:
        popped_node_right = main_huffman_trees.pop(-1)
        right_tree = popped_node_right
    main_huffman_tree = HuffmanTree()
    main_huffman_tree.left = left_tree
    main_huffman_tree.right = right_tree
    main_huffman_tree.number = index
    return main_huffman_tree


def _generate_tree_postorder_version_2(node_lst: list[ReadNode],
                                       root_index: int,
                                       main_huffman_trees: list) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 5, 0, 7), ReadNode(0, 10, 0, 12), \
    ReadNode(1, 0, 1, 0), ReadNode(0, 13, 0, 14), ReadNode(0, 15, 0, 16), \
    ReadNode(1, 0, 1, 0), ReadNode(1,0,1,0)]
    >>> generate_tree_postorder(lst, 6)
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(None,
    HuffmanTree(5, None, None), HuffmanTree(7, None, None)),
    HuffmanTree(None, HuffmanTree(10, None, None),
    HuffmanTree(12, None, None))), HuffmanTree(None,
    HuffmanTree(None, HuffmanTree(13, None, None), HuffmanTree(14, None, None)),
    HuffmanTree(None, HuffmanTree(15, None, None),
    HuffmanTree(16, None, None))))
    """
    node_lst2 = node_lst[:root_index + 1]
    while len(node_lst2) > 0 and root_index > -1:
        node = node_lst2.pop(0)
        merged_tree = _build_huffman_tree_from_readnode(node,
                                                        node_lst.index(node),
                                                        main_huffman_trees)
        main_huffman_trees.append(merged_tree)
        root_index -= 1
    return main_huffman_trees[0]


def generate_tree_postorder(node_lst: list[ReadNode],
                            root_index: int) -> HuffmanTree:
    """ Return the Huffman tree corresponding to node_lst[root_index].
    The function assumes that the list represents a tree in postorder.

    >>> lst = [ReadNode(0, 104, 0, 101), ReadNode(0, 27, 0, 28), \
    ReadNode(1, 0, 0, 114), ReadNode(1, 0, 1, 0), ReadNode(0, 100, 0, 111),
    ReadNode(0, 108, 1, 0), ReadNode(1, 0, 1, 0)]
    >>> generate_tree_postorder(lst, 6)
    HuffmanTree(None, HuffmanTree(None,
    HuffmanTree(None, HuffmanTree(104, None, None),
    HuffmanTree(101, None, None)), HuffmanTree(None,
    HuffmanTree(None, HuffmanTree(27, None, None),
    HuffmanTree(28, None, None)), HuffmanTree(114, None, None))),
    HuffmanTree(None, HuffmanTree(108, None, None), HuffmanTree(None,
    HuffmanTree(100, None, None), HuffmanTree(111, None, None))))
    """
    main_huffman_trees = []
    if len(node_lst) == 0:
        return HuffmanTree()
    return _generate_tree_postorder_version_2(node_lst, root_index,
                                              main_huffman_trees)


def decompress_bytes(tree: HuffmanTree, text: bytes, size: int) -> bytes:
    """ Use Huffman tree <tree> to decompress <size> bytes from <text>.
    >>> f_q = build_frequency_dict(b'a')
    >>> f_q
    {97: 2, 98: 2, 48: 2}
    >>> tree = build_huffman_tree(f_q)
    >>> tree
    HuffmanTree(None, HuffmanTree(48, None, None), HuffmanTree(None,
    HuffmanTree(97, None, None), HuffmanTree(98, None, None)))
    >>> number_nodes(tree)
    >>> code_dict = get_codes(tree)
    >>> code_dict
    {48: '0', 97: '10', 98: '11'}
    >>> compressed_data = compress_bytes(b'a', code_dict)
    >>> list(compressed_data)
    [175, 0]
    >>> decompress_bytes(tree, compressed_data, len(b'a'))
    b'a'
    """
    bits_lst = ''
    for byte in text:
        bits_lst = bits_lst + byte_to_bits(byte)
    bytes_lst = []
    bits_to_symbol_dict = get_codes(tree)
    bits_to_symbol_dict = {y: x for x, y in bits_to_symbol_dict.items()}
    bits = ''
    for bit in bits_lst:
        bits += bit
        if bits in bits_to_symbol_dict:
            bytes_lst.append(bits_to_symbol_dict[bits])
            size -= 1
            bits = ''
        if size == 0:
            break
    return bytes(bytes_lst)


def decompress_file(in_file: str, out_file: str) -> None:
    """ Decompress contents of <in_file> and store results in <out_file>.
    Both <in_file> and <out_file> are string objects representing the names of
    the input and output files.

    Precondition: The contents of the file <in_file> are not empty.
    """
    with open(in_file, "rb") as f:
        num_nodes = f.read(1)[0]
        buf = f.read(num_nodes * 4)
        node_lst = bytes_to_nodes(buf)
        # use generate_tree_general or generate_tree_postorder here
        tree = generate_tree_general(node_lst, num_nodes - 1)
        size = bytes_to_int(f.read(4))
        with open(out_file, "wb") as g:
            text = f.read()
            g.write(decompress_bytes(tree, text, size))


# ====================
# Other functions

def _get_tree_leaf(tree: HuffmanTree, tree_leaf: dict, depth: int = 0) -> \
        dict[int, int]:
    if tree.right is None and tree.left is None:
        if depth in tree_leaf:
            tree_leaf[depth] += 1
        else:
            tree_leaf[depth] = 1
        return tree_leaf
    else:
        if tree.left is not None:
            tree_leaf = _get_tree_leaf(tree.left, tree_leaf, depth + 1)
        if tree.right is not None:
            tree_leaf = _get_tree_leaf(tree.right, tree_leaf, depth + 1)
    return tree_leaf


def improve_tree(tree: HuffmanTree, freq_dict: dict[int, int]) -> None:
    """ Improve the tree <tree> as much as possible, without changing its shape,
    by swapping nodes. The improvements are with respect to the dictionary of
    symbol frequencies <freq_dict>.

    >>> left = HuffmanTree(None, HuffmanTree(97, None, None), \
    HuffmanTree(22, None, None))
    >>> right = HuffmanTree(None, HuffmanTree(39, None, None), \
    HuffmanTree(None, HuffmanTree(10, None, None), HuffmanTree(1, None, None)))
    >>> tree = HuffmanTree(None, left, right)
    >>> freq = {97: 26, 22: 63, 39: 25, 10: 36, 1: 14}
    >>> avg_length(tree, freq)
    2.3048780487804876
    >>> improve_tree(tree, freq)
    >>> tree
    HuffmanTree(None, HuffmanTree(None, HuffmanTree(97, None, None),
    HuffmanTree(22, None, None)), HuffmanTree(None, HuffmanTree(10, None, None),
     HuffmanTree(None, HuffmanTree(39, None, None),
     HuffmanTree(1, None, None))))
    >>> freq2 = freq
    >>> avg_length(tree, freq2)
    2.2378048780487805
    """
    tree_leaf = {}
    tree_leaf = _get_tree_leaf(tree, tree_leaf, 0)
    tree_leaf_symbol = list(freq_dict.items())
    tree_leaf_symbol = sorted(tree_leaf_symbol, key=lambda x: x[1])
    tree_leaf_symbol.reverse()
    tree_leaf_symbol = [x[0] for x in tree_leaf_symbol]
    tree_leaf_symbol_2 = tree_leaf_symbol.copy()
    for depth in tree_leaf:
        tree_leaf[depth], tree_leaf_symbol_2 = tree_leaf_symbol_2[:tree_leaf[
            depth]], tree_leaf_symbol_2[tree_leaf[depth]:]
    nodes = [(tree, 0)]
    while len(nodes) != 0:
        node = nodes.pop(0)
        if node[0].left.symbol is not None:
            if node[0].left.symbol in tree_leaf[node[1] + 1]:
                tree_leaf_symbol.remove(node[0].left.symbol)
            else:
                symbol_substitute = tree_leaf_symbol.pop(0)
                node[0].left.symbol = symbol_substitute
        else:
            nodes.append((node[0].left, node[1] + 1))
        if node[0].right.symbol is not None:
            if node[0].right.symbol in tree_leaf[node[1] + 1]:
                tree_leaf_symbol.remove(node[0].right.symbol)
            else:
                symbol_substitute = tree_leaf_symbol.pop(0)
                node[0].right.symbol = symbol_substitute
        else:
            nodes.append((node[0].right, node[1] + 1))


if __name__ == "__main__":
    import cProfile
    cProfile.run("20+10")

    import doctest

    doctest.testmod()

    import python_ta

    python_ta.check_all(config={
        'allowed-io': ['compress_file', 'decompress_file'],
        'allowed-import-modules': [
            'python_ta', 'doctest', 'typing', '__future__',
            'time', 'utils', 'huffman', 'random'
        ],
        'disable': ['W0401']
    })

    mode = input(
        "Press c to compress, d to decompress, or other key to exit: ")
    if mode == "c":
        fname = input("File to compress: ")
        start = time.time()
        compress_file(fname, fname + ".huf")
        print(f"Compressed {fname} in {time.time() - start} seconds.")
    elif mode == "d":
        fname = input("File to decompress: ")
        start = time.time()
        decompress_file(fname, fname + ".orig")
        print(f"Decompressed {fname} in {time.time() - start} seconds.")
