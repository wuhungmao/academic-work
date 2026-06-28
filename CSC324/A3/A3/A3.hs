{-|
 -
Module:      A3
Description: Assignment 3
Copyright: (c) University of Toronto
               CSC324 Principles of Programming Languages, Fall 2023
-}
-- This lists what this module exports. Don't change this!

module A3 (
    -- Warmup Task 1
    racketifyValue, racketifyExpr,
    -- Warmup Task 2
    cpsFactorial, cpsFibonacci, cpsLength, cpsMap,
    cpsMergeSort, cpsSplit, cpsMerge,
    -- Main Task
    cpsEval
) where

-- You *may not* add imports from Data.Map, or any other imports
import qualified Data.Map (Map, lookup, insert, empty, fromList)
import A3Types (Env, emptyEnv, Value(..), Expr(..))

import Data.List (intercalate)

------------------------------------------------------------------------------
-- * Warmup Task. CPS Transforming Haskell Functions *
------------------------------------------------------------------------------

-- | Compute the factorial of a number
-- factorial :: Int -> Int

-- fact :: Int -> Int
-- fact 0 = 1
-- fact n = n * fact (n - 1)

-- | Compute the factorial of a number, in continuation passing style
cpsFactorial:: Int -> (Int -> r) -> r
cpsFactorial 0 k = k 1
cpsFactorial n k = cpsFactorial (n-1) (\res -> k (n * res))

-- | Compute the n-th fibonacci number F(n).
--    Recall F(0) = 0, F(1) = 1, and F(n) = F(n-1) + F(n-2)

-- fibonacci :: Int -> Int
fibonacci :: Int -> Int
fibonacci 0 = 0
fibonacci 1 = 1
fibonacci n = (fibonacci (n - 1)) + (fibonacci (n - 2))

-- | Compute the n-th fibonacci number F(n), in continuation passing style
cpsFibonacci:: Int -> (Int -> r) -> r
cpsFibonacci 0 k = k 0
cpsFibonacci 1 k = k 1
cpsFibonacci n k = cpsFibonacci (n - 1) ( \res1 -> cpsFibonacci (n - 2) ( \res2 -> k (res1 + res2)))

------------------------------------------------------------------------------
-- | List functions

-- | CPS transform of the function `length`, which computes the length of a list
cpsLength :: [a] -> (Int -> r) -> r
cpsLength [] k = k 0
cpsLength (x:xs) k = cpsLength xs (\res -> k (res + 1))

-- | CPS transform of the function `map`. The argument function (to be applied
--   every element of the list) is written in direct style
cpsMap :: (a -> b) -> [a] -> ([b] -> r) -> r
cpsMap f [] k = k []
cpsMap f (x:xs) k = cpsMap f xs (\res -> k (f x : res))

------------------------------------------------------------------------------
-- Merge Sort

-- | Sort a list using mergeSort
-- mergeSort :: [Int] -> [Int]

-- | Split a list into two lists. All list elements in even indices
-- are placed in one sub-list, and all list elements in odd indices
-- are placed in the second sub-list.
-- split :: [Int] -> ([Int], [Int])

-- | Merge two sorted lists together
-- merge :: [Int] -> [Int] -> [Int]

-- | CPS transform of mergeSort
cpsMergeSort :: [Int] -> ([Int] -> r) -> r
cpsMergeSort [] k = k []
cpsMergeSort [x] k = k [x]
-- split two lst, pass the result to the continuation
-- sort left lst first, and then sort right lst
-- merge two sorted lsts
cpsMergeSort lst k = cpsSplit lst (\(left_lst, right_lst) -> cpsMergeSort left_lst (\res1 -> cpsMergeSort right_lst (\res2 -> cpsMerge res1 res2 k)))

-- | CPS transform of split
cpsSplit :: [Int] -> (([Int], [Int]) -> r) -> r
cpsSplit [] k = k ([], [])
cpsSplit [x] k = k ([x], [])
cpsSplit (x:y:xs) k = cpsSplit xs (\(res1, res2) -> k (x:res1, y:res2)) 

-- | CPS transform of merge
cpsMerge :: [Int] -> [Int] -> ([Int] -> r) -> r
cpsMerge [] lst2 k = k lst2
cpsMerge lst1 [] k = k lst1
cpsMerge (x:xs) (y:ys) k = 
    if x <= y
    then cpsMerge xs (y:ys) (\res -> k (x:res))
    else cpsMerge (x:xs) ys (\res -> k (y:res))

------------------------------------------------------------------------------
-- * Main Task. CPS Transforming The Orange Interpreter *
------------------------------------------------------------------------------

-- | A CPS interpreter `eval` for Orange , which takes an environment,
--   an expression, and a continuation, and calls the continuation with
--   the evaluated value.
--   Notice that the type signature of `eval` is less general compared to
--   what was used above, i.e., it is not:
--      Env -> Expr -> (Value -> r) -> r
--   This restriction on the type of the continuation makes it easier
--   to define `Expr` Haskell data type, and to check for errors.
cpsEval :: Env -> Expr -> (Value -> Value) -> Value

-- eval env (Literal v) = if validLiteral v then v else Error "Literal"
cpsEval env (Literal v) k = 
    if validLiteral v
        then k v 
        else k (Error "Literal")

-- eval env (Plus a b)  = case ((eval env a), (eval env b)) of
-- (Num x, Num y) -> Num (x + y)
-- (Error a, _)   -> Error a
-- (_, Error b)   -> Error b
-- _              -> Error "Plus"
cpsEval env (Plus a b) k = 
    cpsEval env a (\res_a ->
        case res_a of
            (Num a) -> 
                cpsEval env b (\res_b ->
                    case res_b of
                        (Num b) -> k (Num (a + b))
                        (Error e) -> k (Error e)
                        _ -> k (Error "Plus"))
            (Error e) -> k (Error e)
            _ -> k (Error "Plus"))

-- eval env (Times a b) = case ((eval env a), (eval env b)) of
--     (Num x, Num y) -> Num (x * y)
--     (Error c, _)   -> Error c
--     (_, Error d)   -> Error d
--     (c, d)         -> Error "Times"
cpsEval env (Times a b) k =
    cpsEval env a (\res_a ->
        case res_a of
            (Num a) ->
                cpsEval env b (\res_b ->
                    case res_b of
                        (Num b) -> k (Num (a * b))
                        (Error e) -> k (Error e)
                        _ -> k (Error "Times"))
            (Error e) -> k (Error e)
            _ -> k (Error "Times"))

-- eval env (Equal a b) = case ((eval env a), (eval env b)) of
--     (Error c, _)   -> Error c
--     (_, Error d)   -> Error d
--     (c, d)         -> if c == d then T else F
cpsEval env (Equal a b) k =
    cpsEval env a (\res_a ->
        case res_a of
            (Error a) -> k (Error a)
            _ -> cpsEval env b (\res_b ->
                case res_b of
                    (Error b) -> k (Error b)
                    _ -> if res_a == res_b 
                         then k T 
                         else k F))

-- eval env (Cons a b) =  case ((eval env a), (eval env b)) of
--     (Error c, _)   -> Error c
--     (_, Error d)   -> Error d
--     (c, d)         -> Pair c d

cpsEval env (Cons a b) k =
    cpsEval env a (\res_a ->
        case res_a of
            (Error a) -> k (Error a)
            _ -> cpsEval env b (\res_b ->
                case res_b of
                    (Error b) -> k (Error b)
                    _ -> k (Pair res_a res_b)))

-- eval env (First expr) = case (eval env expr) of
--     (Pair a b) -> a
--     (Error c)  -> Error c
--     _          -> Error "First"

cpsEval env (First expr) k
    = cpsEval env expr (\res ->
        case res of
            (Pair a b) -> k a
            (Error c)  -> k (Error c)
            _          -> k (Error "First"))

-- eval env (Rest expr) = case (eval env expr) of
--     (Pair a b) -> b
--     (Error c)  -> Error c
--     _          -> Error "Rest"

cpsEval env (Rest expr) k
    = cpsEval env expr (\res ->
        case res of
            (Pair a b) -> k b
            (Error c)  -> k (Error c)
            _          -> k (Error "Rest"))

-- eval env (If cond expr alt) = case (eval env cond) of
--     (Error c)  -> Error c
--     F          -> (eval env alt)
--     _          -> (eval env expr)

cpsEval env (If cond expr alt) k 
    = cpsEval env cond (\res ->
        case res of
            (Error c) -> k (Error c)
            F         -> cpsEval env alt k
            _         -> cpsEval env expr k)

-- eval env (Var name)  = case (Data.Map.lookup name env) of
--     Just a  -> a
--     Nothing -> Error "Var"

cpsEval env (Var name) k
    = case (Data.Map.lookup name env) of
        Just a  -> k a
        Nothing -> k (Error "Var")

-- TO-DOs: add shift, reset, lambda, app
-- store k into env after binding var to k in env
-- reset k to id
-- From lecture notes:
-- (shift <id> <body>)
-- Bind the current continuation to <id>
-- Evaluates <body> in the current environment
-- ... with one additional binding: <id> is bound to the continuation of the shift expression
-- ... and ignore the continuation of the shift expression
cpsEval env (Shift var expr) k =
  let contfunc = Closure (\[v] k' -> k' (k v))
      env'     = Data.Map.insert var contfunc env
  in cpsEval env' expr id

-- reset just calls the expr with the identity continuation
cpsEval env (Reset expr) k =
  let bodyResult = cpsEval env expr id
  in k bodyResult

cpsEval env (Lambda params body) k_lambda =
  k_lambda $ Closure $ \argvals k_app ->
    if length argvals /= length params
       then Error "Lambda"
       else
         let paramArgTuples = zip params argvals
             newEnv = foldl (\e (param, arg) -> Data.Map.insert param arg e)
                            env
                            paramArgTuples
         in cpsEval newEnv body k_app

cpsEval env (App proc args) k =
  cpsEval env proc (\res_proc ->
    case res_proc of
      (Error e) -> k (Error e)
      (Closure f) ->
        cpsEvalArgs env args (\vargs ->
          f vargs k
        ) k
      _ -> k (Error "App")
  )
  
cpsEval env _ k = undefined
cpsEvalArgs :: Env -> [Expr] -> ([Value] -> Value) -> (Value -> Value) -> Value
cpsEvalArgs _ [] k_args _ = k_args []
cpsEvalArgs env (a:as) k_args k =
  cpsEval env a (\res_a ->
    case res_a of
      (Error e) -> k (Error e)
      _ -> cpsEvalArgs env as (\res_as ->
               k_args (res_a : res_as)) k
  )

-- Helper function (written in direct style) to identify duplicate parameters in a lambda
unique :: (Eq a) => [a] -> [a]
unique [] = []
unique (x:xs)
  | elem x xs = unique xs
  | otherwise = x : unique xs

-- Helper function (written in direct style) to check if a Value contains a Closure/Error
validLiteral :: Value -> Bool
validLiteral T           = True
validLiteral F           = True
validLiteral (Num n)     = True
validLiteral Empty       = True
validLiteral (Pair v w)  = (validLiteral v) && (validLiteral w)
validLiteral (Closure p) = False
validLiteral (Error e)   = False

-- copy from A2
racketifyValue :: Value -> String
racketifyValue T = "#t"
racketifyValue F = "#f"
racketifyValue (Num x) = show x
racketifyValue Empty = "'()"
racketifyValue (Pair a b) = "(cons " ++ racketifyValue a ++ " " ++ racketifyValue b ++ ")"
racketifyValue (Closure _) = error "can't racketify a closure"
racketifyValue (Error _) = error "can't racketify an error value"

racketifyExpr :: Expr -> String
racketifyExpr (Literal v) = racketifyValue v
racketifyExpr (Plus a b) = "(+ " ++ racketifyExpr a ++ " " ++ racketifyExpr b ++ ")"
racketifyExpr (Times a b) = "(* " ++ racketifyExpr a ++ " " ++ racketifyExpr b ++ ")"
racketifyExpr (Equal a b) = "(equal? " ++ racketifyExpr a ++ " " ++ racketifyExpr b ++ ")"
racketifyExpr (Cons a b) = "(cons " ++ racketifyExpr a ++ " " ++ racketifyExpr b ++ ")"
racketifyExpr (First a) = "(car " ++ racketifyExpr a ++ ")"
racketifyExpr (Rest a) = "(cdr " ++ racketifyExpr a ++ ")"
racketifyExpr (Var x) = x
racketifyExpr (If c t f) = "(if " ++ racketifyExpr c ++ " " ++ racketifyExpr t ++ " " ++ racketifyExpr f ++ ")"
racketifyExpr (Lambda xs body) = "(lambda (" ++ intercalate " " xs ++ ") " ++ racketifyExpr body ++ ")"
racketifyExpr (App f xs) = "(" ++ racketifyExpr f ++ " " ++ intercalate " " (map racketifyExpr xs) ++ ")"
racketifyExpr (Shift x e1) = "(shift " ++ x ++ " " ++ racketifyExpr e1 ++ ")"
racketifyExpr (Reset e1) = "(reset " ++ racketifyExpr e1 ++ ")"

-- Comments --
-- Similar to A2, I work with Saabit Zubairi to complete this assignment. His utorid is zubairis.