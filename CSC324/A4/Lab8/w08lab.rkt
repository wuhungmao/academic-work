#lang racket #| * CSC324H5 Week 8 Lab * |#
#|
Module:        w08lab
Description:   Week 8 Lab: Working with the class macro
Copyright: (c) University of Toronto Mississsauga
               CSC324 Principles of Programming Languages, Fall 2023
|#

;-------------------------------------------------------------------------------
; This expression exports functions so they can be imported into other files.
; Don't change it!
(provide Walrus my-class my-class-getter define-data)

(module+ test
  ; Import the testing library
  (require rackunit))

;-------------------------------------------------------------------------------
; * Task 1: Working with objects *
;-------------------------------------------------------------------------------

(define-syntax my-class
  (syntax-rules (method)
    [(my-class <class-name>
               (<attr> ...)
               (method (<method-name> <param> ...) <body>)
               ...)
     (define (<class-name> <attr> ...)
       (lambda (msg)
         (cond [(equal? msg (quote <attr>)) <attr>]
               ...
               [(equal? msg (quote <method-name>))
                (lambda (<param> ...) <body>)]
               ...
               [else "Unrecognized message!"])))]))

; TODO define the class Walrus here
(my-class Walrus (name age bucket)
          (method (bucket-empty?) (empty? bucket))
          (method (same-name? other) (equal? (other 'name) name))
          (method (equal? other) (and (equal? (other 'name) name) (equal? (other 'age) age) ))
          (method (catch item) (Walrus name age (cons item bucket)))
          (method (have-birthday) (Walrus name (+ age 1) (cons "cake" bucket)))
          )

(module+ test
  (define alice (Walrus "alice" 4 '()))
  (define bob (Walrus "alice" 4 '()))

  (test-true "alice and bob are both functions"
             (and (procedure? alice) (procedure? bob)))
  (test-true "alice and bob have different names"
             ((alice 'same-name?) bob))
  ; TODO: add more tests here!
  )


;-------------------------------------------------------------------------------
; * Task 2: Accessor function in `my-class` *
;-------------------------------------------------------------------------------

#|
(my-class-getter <Class> (<attr> ...)
  (method (<method-name> <param> ...) <body>) ...)

  This macro accepts the *exact* same pattern as my-class from above.

  In addition to defining the constructor, my-class-getter defines
  *one new accessor function per attribute of the class*, using the name
  of the attribute as the name of the function.

  Implementation notes:
    - Our starter code has wrapped the `define` from lecture inside a `begin`.
      This is required so that you can add more `define`s after the one for the
      constructor.
|#
(define-syntax my-class-getter
  (syntax-rules (method)
    [(my-class-getter <class-name>
                      (<attr> ...)
                      (method (<method-name> <param> ...) <body>)
                      ...)
     (begin
       (define (<class-name> <attr> ...)
         (lambda (msg)
           (cond [(equal? msg (quote <attr>)) <attr>]
                 ...
                 [(equal? msg (quote <method-name>))
                  (lambda (<param> ...) <body>)]
                 ...
                 [else "Unrecognized message!"])))
       (define (<attr> obj) (obj (quote <attr>)))
       ...
       )]))

(module+ test
  ; We use `local` to create a local scope for our definitions.
  ; Run these tests when you're ready!
  (local
    [(my-class-getter Point (x y))]
    (test-true "x and y are functions" (and (procedure? x) (procedure? y)))
    (test-equal? "x and y are accessors"
                 (let ([p (Point 2 3)])
                   (list (x p) (y p)))
                 (list 2 3))))

;-------------------------------------------------------------------------------
; * Task 3: Simplified Algebraic Data Type Declarations *
;-------------------------------------------------------------------------------

#|
This macro is a simplified implementation of algebraic data type
declaration that we saw in Haskell.

This macro will create all the constructors for the associated data types.
Each constructor will take the appropriate arguments and produce a list
consisting of the type, the constructor name, and the arguments.
For example, calling the constructor `(MyPoint 3 4)` should produce the
list `'(Point MyPoint 3 4)`.
|#

; my solution
(define-syntax define-data
  (syntax-rules ()
    [(define-data <type> (<constructor-name> <args> ...) ...) ; TODO: change this pattern
     (begin
       (define (<constructor-name> <args> ...)
         (list (quote <type>) (quote <constructor-name>) <args> ...)
         )...
          )]))

; (define (MyPoint x y)
;   (Point MyPoint x y)
;   )
(module+ test
  ; We use `local` to create a local scope for our definitions.
  ; Run these tests when you're ready!
  (local
    [(define-data Point (MyPoint x y))
     (define-data Shape (Circle p r) (Rectangle p1 p2))]
    (test-true "MyPoint is a function" (procedure? MyPoint))
    (test-equal? "(MyPoint 3 4) produces the correct output"
                 (MyPoint 3 4)
                 '(Point MyPoint 3 4))
    (test-true "Circle is a function" (procedure? Circle))
    (test-true "Rectangle is a function" (procedure? Rectangle))
    (test-equal? "(Circle (MyPoint 3 4) 5) produces the correct output"
                 (Circle (MyPoint 3 4) 5)
                 '(Shape Circle (Point MyPoint 3 4) 5))
    (test-equal? "(Rectangle (MyPoint 1 2) (MyPoint 3 4)) produces the correct output"
                 (Rectangle (MyPoint 1 2) (MyPoint 3 4))
                 '(Shape Rectangle (Point MyPoint 1 2) (Point MyPoint 3 4))))
  )



