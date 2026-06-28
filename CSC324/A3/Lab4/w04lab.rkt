#lang racket #| * CSC324H5 Fall 2024: Week 4 Lab * |#
#|
Module:        w04lab
Description:   Week 4 Lab: Functional Data Structures
Copyright: (c) University of Toronto Mississsauga
               CSC324 Principles of Programming Languages, Fall 2024
|#

; This specifies which functions this module exports. Don't change this!
(provide execute)

; NOTE: As is common to testing frameworks, by default DrRacket only displays
; output for *failing* tests. If you run the module with the tests uncommented
; but don't see any output, that's good---the tests all passed! (If you want
; to double-check this, you can try breaking test cases and seeing the "fail"
; output yourself.)
(module+ test
  ; Import the testing library
  (require rackunit))

;-------------------------------------------------------------------------------

(define/match (set_get lst ht)

  ;; General Case: lst is non-empty.
  [(lst hash_table)
    
    (match lst
      ;; 1.1 'get' operation
      [(list (list 'get id) rest ...) 
       (cons (hash-ref hash_table id 'error)
             (set_get rest hash_table))]

      [(list 'get id) 
       (list (hash-ref hash_table id 'error))]
      
      ;; 1.2 'set' operation
      [(list (list 'set id val) rest ...) 
       (set_get rest (hash-set hash_table id val))]

      [(list 'set id val) 
       ((hash-set hash_table id val))]

      ['()
       '()]
      
      ;; Fallback for non-empty but unmatchable list (optional, but helpful)
      [_ (error 'set_get "Unrecognized command list structure: ~v" lst)]
      )
    ]
      
  ;; Base Case: lst is empty.
  
)
#|
(execute cmds) -> list? 
  cmds: list?
    A list of "get" and "set" commands

  Returns a list of resulting numbers from the "get" commands
|#
(define (execute lst)
  (set_get lst (hash)))


(module+ test
  (test-equal? "execute: set and get a value"
               (execute '((set a 3) (get a)))
               '(3))
  (test-equal? "execute: set a value multiple times"
               (execute '((set a 3) (set a 4) (get a)))
               '(4))
  (test-equal? "execute: test from the handout"
               (execute '((set a 3) (set b 4) (get a) (set b 5) (get b) (get c)))
               '(3 5 error))
)

