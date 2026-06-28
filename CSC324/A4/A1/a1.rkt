#lang racket #| * CSC324H5 Fall 2024: Assignment 1 * |#
#|
Module:        a1
Description:   Assignment 1: Syntactic Sugar
Copyright: (c) University of Toronto Mississsauga
               CSC324 Principles of Programming Languages, Fall 2024
|#

; This specifies which functions this module exports. Don't change this!
(provide desugar)

; Import the testing library
(module+ test
  (require rackunit))

;;; Desugaring Addition
;;; A MandarinSugar expression
;;; (+ a b c d ...)
;;; should desugared into the expression
;;; (+ a (+ b (+ c (+ d ...))))
;;; Note: Here, we will assume that a, b, c, . . . etc are just identifiers. If a, b, c, . . . are themselves MandarinSugar
;;; expressions, then those expressions will also need to be desugared.

;;; use it as helper function
(define/match (let_helper list_identifier_expr_pair)
  ;; The helper function is defined inside the main function.
  ;; It uses an accumulator `acc` to keep track of the running total.

  [((list (list identifier expr) rest))
  ;;;  (displayln '"let helper ")
  ;;;  (displayln (string-append "identifier " (format "~s" identifier)))
  ;;;  (displayln (string-append "expr: " (format "~s" expr)))
  ;;;  (displayln (string-append "rest: " (format "~s" rest)))
   (list (list identifier (desugar expr)) (let_helper rest))]
  
  [((list identifier expr))
  ;;;  (displayln '"let helper 2")
  ;;;  (displayln (string-append "identifier " (format "~s" identifier)))
  ;;;  (displayln (string-append "expr: " (format "~s" expr)))
   (list identifier (desugar expr))]
  [('()) '()]
  )

;; A function that sums the elements of a list using a helper function.
(define (const_match_input val poss_val)
  ;; The helper function is defined inside the main function.
  ;; It uses an accumulator `acc` to keep track of the running total.
  ;;; (displayln (string-append "--------------------- " (format "~s" '-)))
  ;;; (displayln (string-append "val in const_match_input: " (format "~s" val)))
  ;;; (displayln (string-append "poss_val in const_match_input: " (format "~s" poss_val)))
  ;;; (displayln (string-append "return from const_match_input: " (format "~s" (cons 'match (cons val poss_val)))))
  ;;; (displayln (string-append "--------------------- " (format "~s" '-)))
  (cons 'match (cons val poss_val))
  )

(define/match (desugar prog)

  [((list 'match val (list poss_val expr) (list '_ expr2)))
  ;;;  (displayln (string-append "match to bottom " (format "~s" '-)))
  ;;;  (displayln (string-append "val: " (format "~s" val)))
  ;;;  (displayln (string-append "poss_val: " (format "~s" poss_val)))
  ;;;  (displayln (string-append "expr: " (format "~s" expr)))
   ;;; no need to decompose
   (cond
     ;;; possible value is a number
     [(number? poss_val) (list 'if (list '= val poss_val) expr expr2)] 
     ;;; possible value is a variable
     [(symbol? poss_val) (list 'if '#t (list 'let (list (list poss_val val)) expr) expr2)] 
     ;;; possible value is a pair
     [(pair? poss_val)
      (match poss_val
        [(list 'cons head tail)
        ;;;  (displayln (string-append "head: " (format "~s" head)))
        ;;;  (displayln (string-append "tail: " (format "~s" tail)))
         (list 'if (list 'pair? val) (list 'let (list (list head (list 'car val))  (list tail (list 'cdr val))) (desugar expr)) expr2)]
        )]
     )
   ]

  [((list 'match val (list poss_val expr) poss_val2 ...))
  ;;;  (displayln (string-append "match to top " (format "~s" '-)))
  ;;;  (displayln (string-append "val: " (format "~s" val)))
  ;;;  (displayln (string-append "poss_val: " (format "~s" poss_val)))
  ;;;  (displayln (string-append "expr: " (format "~s" expr)))
  ;;;  (displayln (string-append "poss_val2: " (format "~s" poss_val2)))
   ;;;  (displayln (string-append "expr2: " (format "~s" expr2)))

   ;;;  (if (> (length poss_val2) 1)
   ;;; need to decompose
   (cond
     ;;; possible value is a number
     [(number? poss_val) (list 'if (list '= val poss_val) (desugar expr) (desugar (const_match_input val poss_val2)))] 
     ;;; possible value is a variable
     [(symbol? poss_val) (list 'if '#t (list 'let (list (list poss_val val)) (desugar expr)) (desugar (const_match_input val poss_val2)))] 
     ;;; possible value is a pair
     [(pair? poss_val)
      (match poss_val
        [(list 'cons head tail)
        ;;;  (displayln (string-append "head: " (format "~s" head)))
        ;;;  (displayln (string-append "tail: " (format "~s" tail)))
         (list 'if (list 'pair? val) (list 'let (list (list head (list 'car val))  (list tail (list 'cdr val))) (desugar expr)) (desugar (const_match_input val poss_val2)))]
        )]
     )
   ]

  ;;; (let ((IDENTIFIER <MandarinSugarExpr>) ...) <MandarinSugarExpr>) ; let expressions
  [((list 'let (list list_identifier_expr_pair ...) expr2)) 
  ;;;   (displayln '"let statement ")
  ;;;  (displayln (string-append "list_identifier_expr_pair: " (format "~s" list_identifier_expr_pair)))
  ;;;  (displayln (string-append "expr2 " (format "~s" expr2)))
   (list 'let  (let_helper list_identifier_expr_pair) (desugar expr2))]

  ;;; (if <MandarinBasicExpr> <MandarinBasicExpr> <MandarinBasicExpr>)
  [((list 'if expr1 expr2 expr3))
  ;;;  (displayln '"if statement ")
  ;;;  (displayln (string-append "expr1: " (format "~s" expr1)))
  ;;;  (displayln (string-append "expr2: " (format "~s" expr2)))
  ;;;  (displayln (string-append "expr3: " (format "~s" expr3)))
   (list 'if (desugar expr1) (desugar expr2) (desugar expr3))]

  ;;; (cons <MandarinSugarExpr> <MandarinSugarExpr>) ; create pairs
  [((list 'cons expr1 expr2)) 
  ;;;   (displayln '"cons statement ")
  ;;;  (displayln (string-append "expr1: " (format "~s" expr1)))
  ;;;  (displayln (string-append "expr2: " (format "~s" expr2)))
  (list 'cons (desugar expr1) (desugar expr2))]

  ;;; (car <MandarinSugarExpr>) first
  [((list 'car expr1)) 
  ;;;  (displayln '"car statement ")
  ;;;  (displayln (string-append "expr1: " (format "~s" expr1)))
  (list 'car (desugar expr1))]

  ;;; (cdr <MandarinSugarExpr>) rest
  [((list 'cdr expr1)) 
  ;;;   (displayln '"cdr statement ")
  ;;;  (displayln (string-append "expr1: " (format "~s" expr1)))
  (list 'cdr (desugar expr1))]

  ;;; (pair? <MandarinSugarExpr>)
  [((list 'pair? expr1)) 
  ;;;   (displayln '"pair? statement ")
  ;;;  (displayln (string-append "expr1: " (format "~s" expr1)))
  (list 'pair? (desugar expr1))]

  ;;; (= <MandarinSugarExpr> <MandarinSugarExpr>) ; equality testing
  [((list '= expr1 expr2)) 
  ;;;   (displayln '"= statement ")
  ;;;  (displayln (string-append "expr1: " (format "~s" expr1)))
  ;;;  (displayln (string-append "expr2: " (format "~s" expr2)))
  (list '= (desugar expr1) (desugar expr2))]

  [((list 'lambda (list expr_id4) expr_id5))
  ;;;   (displayln '"lambda statement ")
  ;;;  (displayln (string-append "expr_id4: " (format "~s" expr_id4)))
  ;;;  (displayln (string-append "expr_id5: " (format "~s" expr_id5)))
   (list 'lambda (list (desugar expr_id4)) (desugar expr_id5))
   ]

  [((list 'lambda (list expr_id expr_id2 ...) expr_id3))
  ;;;   (displayln '"lambda statement 2")

  ;;;  (displayln (string-append "expr_id: " (format "~s" expr_id)))
  ;;;  (displayln (string-append "expr_id2: " (format "~s" expr_id2)))
  ;;;  (displayln (string-append "expr_id3: " (format "~s" expr_id3)))
   (list 'lambda (list expr_id) (desugar (list 'lambda expr_id2 expr_id3)))
   ]

  [((list '+ expr_id7))
   ;;;(displayln (string-append "expr_id6: " (format "~s" expr_id6)))
  ;;;   (displayln '"+ statement ")
  ;;;  (displayln (string-append "expr_id7: " (format "~s" expr_id7)))
   (list (desugar expr_id7))]

  [((list '+ expr_id7 expr_id8))
  ;;;   (displayln '"+ statement 2")

  ;;;  ;;;(displayln (string-append "expr_id6: " (format "~s" expr_id6)))
  ;;;  (displayln (string-append "expr_id7: " (format "~s" expr_id7)))
  ;;;  (displayln (string-append "expr_id8: " (format "~s" expr_id8)))
   (list '+ (desugar expr_id7) (desugar expr_id8))]

  [((list '+ expr_id2 expr_id3 ...))
  ;;;   (displayln '"+ statement 3")
  ;;;  ;;;(displayln (string-append "expr_id: " (format "~s" expr_id)))
  ;;;  (displayln (string-append "expr_id2: " (format "~s" expr_id2)))
  ;;;  (displayln (string-append "expr_id3: " (format "~s" expr_id3)))
   (list '+ (desugar expr_id2) (desugar (cons '+ expr_id3)))]

  ;;; n-ary function creation
  [((list func arg arg2 ...))
  ;;;   (displayln '"func statement ")

  ;;;  (displayln (string-append "func: " (format "~s" func)))
  ;;;  (displayln (string-append "arg " (format "~s" arg)))
  ;;;  (displayln (string-append "arg2 " (format "~s" arg2)))
   (cons (desugar func) (cons (desugar arg) (desugar arg2)))]

  [(prog)
  ;;;  (displayln (string-append "prog: " (format "~s" prog)))
   prog]

  [('()) '()]
  )

;;; (desugar '(lambda (a b c) (+ a b c)))
; You can write helper functions freely
(module+ test

  ;;; handle the case of n-ary function calls

  ;;;(test-equal? "Desugaring n-ary function calls, with nested MandarinSugar structures in the function body"
  ;;;(desugar_addition '((lambda (a b c) (+ a b c)) 1 2 3))
  ;;;'((((lambda (a) (lambda (b) (lambda (c) (+ a (+ b c))))) 1) 2) 3))

  ; We use rackunit's test-equal? to define some simple tests.
  ;;; handle the case of addition
  (test-equal? "Desugaring identifier" 
               (desugar 'a)             
               'a)                      
  (test-equal? "Desugaring n-ary function calls" 
               (desugar '(f a b))             
               '(f a b))
  (test-equal? "Desugaring n-ary function calls" 
               (desugar '(f a b c))             
               '(f a b c))
  (test-equal? "Desugaring if expression" 
               (desugar '(if (= 2 2) 
                             "numbers are equal"
                             "numbers are not equal"))             
               '(if (= 2 2)
                     "numbers are equal"
                     "numbers are not equal")
               )
  
  (test-equal? "Desugaring n-ary function calls, with nested MandarinSugar structures in the function body"
               (desugar '(lambda (a b c) (+ a b c)))
               '(lambda (a) (lambda (b) (lambda (c) (+ a (+ b c))))))
  (test-equal? "Desugaring a constant" ; Test label
               (desugar 3)             ; Actual value
               3)                      ; Expected value
  (test-equal? "Desugaring binary addition"
               (desugar '(+ 3 3))
               '(+ 3 3))
  (test-equal? "Desugaring binary addition"
               (desugar '(+ 3))
               '(3))
  (test-equal? "Desugaring n-ary addition"
               (desugar '(+ 1 2 3))
               '(+ 1 (+ 2 3)))
  (test-equal? "Desugaring n-ary addition"
               (desugar '(+ 1 2 3 4))
               '(+ 1 (+ 2 (+ 3 4))))
  (test-equal? "Desugaring let expression"
               (desugar '(let ((x 10)
                               (y 20))
                           (+ x y)))
               '(let ((x 10)
                      (y 20))
                  (+ x y)))
  (test-equal? "Desugaring cons expression"
               (desugar '(cons a b))
               '(cons a b))
  (test-equal? "Desugaring car expression"
               (desugar '(car 1))
               '(car 1))
  (test-equal? "Desugaring cdr expression"
               (desugar '(cdr 1))
               '(cdr 1))
  (test-equal? "Desugaring pair expression"
               (desugar '(pair? (cons 1 2)))
               '(pair? (cons 1 2)))
  (test-equal? "Desugaring equal expression"
               (desugar '(= 5 5))
               '(= 5 5))
  (test-equal? "Desugaring match value"
               (desugar '(match x
                           (1 5)
                           (_ 6)))
               '(if (= x 1)
                    5
                    6)
               )
  (test-equal? "Desugaring match identifier"
               (desugar '(match x
                           (a (+ a 1))
                           (_ 2)))
               '(if #t
                    (let ((a x))
                      (+ a 1))
                    2)
               )
  (test-equal? "Desugaring match pair"
               (desugar '(match lst
                           ((cons x xs) (+ x 1))
                           (_ 0)))
               '(if (pair? lst)
                    (let ((x (car lst))
                          (xs (cdr lst)))
                      (+ x 1))
                    0)
               )

  (test-equal? "Desugaring match 2"
               (desugar '(match x
                           (1 expr1)
                           ((cons a d) expr2)
                           (_ expr4)
                           ))
               '(if (= x 1) ; literal pattern 1 matches if x = 1
                    expr1
                    (if (pair? x) ; cons pattern matches if x is a pair
                        ; we must bind a and d to their appropriate values
                        (let ((a (car x)) (d (cdr x)))
                          expr2)
                        expr4
                        ))
               )
               
  (test-equal? "Desugaring match multiple"
               (desugar '(match x
                           (1 expr1)
                           ((cons a d) expr2)
                           (y expr3)
                           (_ expr4)))
               '(if (= x 1) ; literal pattern 1 matches if x = 1
                    expr1
                    (if (pair? x) ; cons pattern matches if x is a pair
                        ; we must bind a and d to their appropriate values
                        (let ((a (car x)) (d (cdr x)))
                          expr2)
                        (if #t ; an identifier pattern always matches
                            ; we must bind y to its appropriate value
                            (let ((y x))
                              expr3)
                            expr4)))
               )
  )

