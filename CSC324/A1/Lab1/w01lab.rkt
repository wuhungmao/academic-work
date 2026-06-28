#lang racket #| * CSC324H5 Fall 2024: Week 1 Lab * |#
#|
Module:        w01lab
Description:   Week 1 Lab Introduction to Racket and Haskell
Copyright: (c) University of Toronto Mississsauga
               CSC324 Principles of Programming Languages, Fall 2024

In this part of the exercise, you'll get started writing some simple
functions in Racket. Since this is likely your first time using Racket,
we strongly recommend going through some of the documentation we listed
under the "Software" page as you work through this exercise.
In comments below, we also give some links to documentation to built-in
functions for standard data types (numbers, strings, lists) that we want
you to become familiar with.

Finally, you'll notice the (module+ test ...) expressions interleaved with
the function definitions; this is a standard Racket convention for simple
unit tests that we'll use throughout the course. Please read them carefully,
and add tests of your own!
|#

; This specifies which functions this module exports. Don't change this!
(provide percentage
         n-copies
         appears
         calculate)

; We use (module+ test ...) to mark code that shouldn't be evaluated when this
; module is imported, but instead is used for testing purposes (similar to Python's
; if __name__ == '__main__').
;
; Right now each module+ expression after this one (i.e., the ones that actually
; contain test cases) is commented out, because they all fail, and DrRacket runs
; tests automatically when you press "Run".
; As you do your work, uncomment the module+ expression by deleting the `#;` in
; front of the module+ expressions and run the module to run the tests.
;
; NOTE: As is common to testing frameworks, by default DrRacket only displays
; output for *failing* tests. If you run the module with the tests uncommented
; but don't see any output, that's good---the tests all passed! (If you want
; to double-check this, you can try breaking test cases and seeing the "fail"
; output yourself.)
(module+ test
  ; Import the testing library
  (require rackunit))

;-------------------------------------------------------------------------------

#|
(percentage decimal) -> integer?
  decimal: number?
    The decimal number to be converted into a percentage.

  Returns the percentage representation of the decimal number, rounded to
  the nearest integer.

  Note: number? is a predicate that returns whether its argument of a numeric type.
  The line "percent: number?" is analogous to a type annotation, and specifies
  that the argument for `percent` should always be a number. Similarly, the "-> integer?"
  at the end of the first line indicates that this function should return an integer.
  We'll illustrate more of this style of Racket documentation---called "contracts"---
  throughout this exercise and beyond.

  Relevant documentation: https://docs.racket-lang.org/reference/generic-numbers.html.

  Use the `round` function for rounding.
|#
(define (percentage decimal)
  ; TODO: replace the (void) with a proper function body.
  (round (let ([percentage (* decimal 100)]) percentage)))

(module+ test
  ; We use rackunit's test-equal? to define some simple tests.
  (test-equal? "(percentage 0.2023)"  ; Test label
               (percentage 0.2023)    ; Actual value
               20.0)                  ; Expected value
  (test-equal? "(percentage 1.25555)"
               (percentage 1.25555)
               126.0))

;-------------------------------------------------------------------------------

#|
(n-copies s n) -> string?
  s: string?
  n: (and/c integer? (not/c negative?))

  Returns a string consisting of n copies of s.
  *Use recursion!* Remember that you aren't allowed to use mutation for this exercise.

  Note: `and/c` and `not/c` are operators on predicates; the line
  `n: (and/c integer? (not/c negative?))` means that the argument `n` must be an
  integer and cannot be negative (so n >= 0).

  Relevant documentation: https://docs.racket-lang.org/reference/strings.html
|#
(define (n-copies s n)
  ; TODO: replace the (void) with a proper function body.
  (if (> n 0)
    (string-append s (n-copies s (- n 1)))
    ""
  )
)

(module+ test
  (test-equal? "n-copies: Three copies"
               (n-copies "Hello" 3)
               "HelloHelloHello")
  (test-equal? "n-copies: Zero copies"
               (n-copies "Hello" 0)
               "")
  (test-equal? "n-copies: Single letter"
                (n-copies "a" 10)
                "aaaaaaaaaa"))

;-------------------------------------------------------------------------------

#|
(appears s l) -> bool?
  s: symbol?
  l: list?

  Returns whether s appears inside the list l
  Use recursion! Do not use any list functions other than `first` and `rest`
  Once again, remember that you aren't allowed to use mutation in this function.

  Relevant documentation: https://docs.racket-lang.org/reference/pairs.html
|#
(define (appears s l)
  (cond [(empty? l)
         ; In this case the list is empty. What should be returned?
         #f]
        [else
         ; In this case the list is non-empty. Divide the list into its first
         ; element and the remaining elements in the list, recurse on the remaining,
         ; and combine the results in some way to return the final result.
         (let* ([first-element   (first l)]
                [the_rest  (appears s (rest l))]) ; Pick a better variable name.
           (if(equal? first-element s)
              #t
              the_rest
            )
           )])
  )

(module+ test
  (test-equal? "appears: Element appears is in the first position in the list"
               (appears 'apple (list 'apple 'banana 'orange))
               #t)
  (test-equal? "appears: Element appears is in the second position in the list"
               (appears 'banana (list 'apple 'banana 'orange))
               #t)
  (test-equal? "appears: Element does not appear in the list"
               (appears 'csc324 (list 'apple 'banana 'orange))
               #f))

;-------------------------------------------------------------------------------

#|
(calculate expr)
  expr: An expression generated by the Binary Arithmetic Expression Grammar
        described in the handout.

  Return the numerical value of the expression
|#
(define (calculate expr)
  (if (number? expr)
    expr
    (
      ;;; it is list    
        (cond 
        [(equal? (first expr) '+) +]
        [(equal? (first expr) '-) -]
        [(equal? (first expr) '*) *]
        [(equal? (first expr) '/) /]
        )
      
        (if (list? (second expr))
          (calculate (second expr))
          (second expr)
        ) 

        (if (list? (third expr))
          (calculate (third expr))
          (third expr)
        )
      ;;; )
    )
  )
)

; uncomment this when you are ready
(module+ test
  (test-equal? "calculate: +"
               (calculate '(+ 2 3));'(+ 2 3) is the same as (list '+ 2 3) 
               5)
  (test-equal? "calculate: /"
               (calculate '(/ (+ 2 6) 2))
               4))

