#lang racket #| * CSC324H5 Fall 2024: Week 2 Lab * |#
#|
Module:        w02lab
Description:   Week 2 Lab: Pattern Matching and Recursion
Copyright: (c) University of Toronto Mississsauga
               CSC324 Principles of Programming Languages, Fall 2024
|#

; This specifies which functions this module exports. Don't change this!
(provide replace-all
         replace-all-helper
         subst)

; NOTE: As is common to testing frameworks, by default DrRacket only displays
; output for *failing* tests. If you run the module with the tests uncommented
; but don't see any output, that's good---the tests all passed! (If you want
; to double-check this, you can try breaking test cases and seeing the "fail"
; output yourself.)
(module+ test
  ; Import the testing library
  (require rackunit))

;-------------------------------------------------------------------------------
; Task 1

#|
(replace-all lst id with) -> list?
  lst: list? 
    A list of values
  id: symbol? 
    A value to be replaced in the list
  with: symbol? 
    A value that will take the place of `id` in `lst`

  Returns a list of symbols, almost identical to `lst`, but with every
  instance of `id` replaced with `with`
|#
(define/match (replace-all lst id with)
  ; the base case 
  [('() id with) ; What pattern should go in TODO?
   '()]
  ; the recursive case
  [((cons x xs) id with) ; What pattern should go in TODO?
   (
    cons 
    (if (equal? x id)
        with
        x)
    (replace-all xs id with)
   )])       ; What code should go here?

#|
(replace-all-tail lst id with) -> list?
  lst: list? 
    A list of values
  id: symbol? 
    A value to be replaced in the list
  with: symbol? 
    A value that will take the place of `id` in `lst`

  Returns a list of symbols, almost identical to `lst`, but with every
  instance of `id` replaced with `with`. This function should be
  tail recursive.

  This function is written for you
|#
(define (replace-all-tail lst id with)
    (replace-all-helper lst id with '()))

#|
(replace-all-helper lst id with acc) -> list?
  lst: list? 
    A list of values
  id: symbol? 
    A value to be replaced in the list
  with: symbol? 
    A value that will take the place of `id` in `lst`
  acc: list?
    An accumulator

  Helper function to replace-all-tail.
|#

(define/match (replace-all-helper lst id with acc)
  [('() id with acc) (reverse acc)
  ]
  [((cons x xs) id with acc) 
    (if (equal? x id)
        (replace-all-helper xs id with (cons with acc))
        (replace-all-helper xs id with (cons x acc))
    )
  ]
)

(module+ test
  (test-equal? "replace-all: empty list"
               (replace-all '() 'a 'b)
               '())
  (test-equal? "replace-all: some replacement"
               (replace-all '(b a c a) 'a 'b)
               '(b b c b))
  ; TODO: write tests for replace-all-tail or replace-all-helper to test your code
  (test-equal? "replace-all-tail: some replacement"
               (replace-all-tail '(b a c a) 'a 'b)
               '(b b c b))
)

;-------------------------------------------------------------------------------
; Task 2

#|
(subst expr id with) -> (and/c symbol? list? )
  expr: (and/c symbol? list?) 
    An expression following the grammar provided in the handout
  id:   symbol?
    A symbol to replace
  with: (and/c symbol? list?)
    An expression following hte grammar provided in the handout, to replace
    `id` with.

  Recursively replace any instance of `id` in `expr` with the sub-expression
  `with`. However, 
|#
(define/match (subst expr id with)
  [((list 'lambda (list param) body) id with) ; write pattern for lambda expressions
   (if(symbol? body)
      (if(equal? body id)
         (list 'lambda (list param) with)
         (list 'lambda (list param) body))
      (list 'lambda (list param) (subst body id with)
            ))
   ]
  [((list 'f expr) id with) ; write pattern for function calls
   (if(symbol? expr)
      (if(equal? expr id)
         (list 'f with)
         (list 'f expr)
         )
      (list 'f (subst expr id with)))           ; place holder for code to handle function calls
   ]
  [((list expr1 expr2) id with) ; write simple pattern for identifiers: any expression that
   ; didn't match the previous patterns should match here
   (list (subst expr1 id with) (subst expr2 id with))
   ]
  [(expr id with) ; write simple pattern for identifiers: any expression that
   ; didn't match the previous patterns should match here
   (if(equal? expr id)
      with
      expr
      )       ; place holder for code to handle function calls
   ]

  )

(module+ test
  ; We use rackunit's test-equal? to define some simple tests.
  (test-equal? "subst: a single symbol"  ; Test label
               (subst 'a 'a 'b)          ; Actual value
               'b)                       ; Expected value
  (test-equal? "subst: a function call"
               (subst '(f a) 'a '(g b))
               '(f (g b)))

  (test-equal? "subst: a lambda"
               (subst '(a b) 'a 'c)
                '(c b))
  (test-equal? "subst: nested function calls"
               (subst '(f (a (c a))) 'a 'b)
               '(f (b (c b))))
  (test-equal? "subst: lambdas in nested function calls"
               (subst '(a ((lambda (a) (f a)) a)) 'a 'b)
               '(b ((lambda (a) (f a)) b)))
  (test-equal? "subst: lambdas in nested function calls 2"
               (subst '(b ((lambda (c) (f a)) a)) 'a 'b)
               '(b ((lambda (c) (f b)) b)))

  (test-equal? "subst: b lambda"
               (subst 'b 'a 'c)
                'b)
  (test-equal? "subst: b lambda"
               (subst (((lambda (a) a)) 'a) 'a 'c)
                '((lambda (a) c) c))

  (test-equal? "subst: b lambda"
               (subst 'b 'a 'c)
                'b)
  (test-equal? "subst: b lambda"
               (subst ('b (lambda (a) ('b 'b))) 'a 'c)
                ('b (lambda (a) ('b 'b)))) 

  (test-equal? "subst: a function call"
               (subst ((lambda (b) 'a) (lambda (a) 'a)) 'a 'c)
               '((lambda (b) 'c) (lambda (a) 'c)))

    
)

