-- This lists what this module exports. Don't change this!

-- |
-- Module: A2
-- Description: Assignment 2
-- Copyright: (c) University of Toronto Mississagua
--                CSC324 Principles of Programming Languages, Fall 2025
module A2
  ( run,
    eval,
  )
where

-- You *may not* add imports from Data.Map, or any other imports
import A2Types(Expr(..), Value(..), Env)
import qualified Data.Map (lookup, insert, empty)
import Data.List (intercalate)

-- | Runs an Orange expression by calling `eval` with the empty environment
run :: Expr -> Value
run e = eval Data.Map.empty e

-- Explanation:
-- 'run' is a convenience function that evaluates an expression
-- in a completely empty environment (no variables defined yet).

-- Explanation:
-- 'run' is a convenience function that evaluates an expression
-- in a completely empty environment (no variables defined yet).

-- | An interpreter for the Orange language.
eval :: Env -> Expr -> Value
-- Evaluate a literal value
eval env (Literal v) =
  case v of
    Num x -> Num x -- Numbers evaluate to themselves
    T -> T -- Boolean true evaluates to itself
    F -> F -- Boolean false evaluates to itself
    Empty -> Empty -- Empty list/value evaluates to itself
    Pair a b -> Pair a b -- Pairs evaluate to themselves (no recursive evaluation here)
    Closure args env body -> Closure args env body -- Closures are returned as-is
    Error msg -> Error msg -- Errors propagate as-is

-- Evaluate addition
eval env (Plus a b) = case ((eval env a), (eval env b)) of
  (Num x, Num y) -> Num (x + y) -- Add numbers if both arguments evaluate to numbers
  (Error x, _) -> Error x
  (_, Error y) -> Error y
  _ -> Error "Plus" -- Type error if either is not a number
  -- Note: no other patterns are missing here; any non-number case is already caught by '_'

-- Evaluate multiplication
eval env (Times a b) = case ((eval env a), (eval env b)) of
  (Num x, Num y) -> Num (x * y) -- Multiply numbers if both arguments evaluate to numbers
  (Error x, _) -> Error x
  (_, Error y) -> Error y
  _ -> Error "Times" -- Type error if either is not a number
  -- Note: no other patterns are missing here; any non-number case is already caught by '_'

-- Evaluate equal
eval env (Equal a b) = case ((eval env a), (eval env b)) of
  (Error x, _) -> Error x
  (_, Error y) -> Error y
  (result1, result2) -> if result1 == result2 then T else F
  _ -> (Error "Equal")

-- Evaluate cons
eval env (Cons a b) = case (eval env a, eval env b) of
  (Error x, _) -> Error x
  (_, Error y) -> Error y
  (result1, result2) -> Pair result1 result2
  _ -> (Error "Cons")

-- Evaluate First
eval env (First a) = case ((eval env a)) of
  (Error x) -> Error x
  (Pair result1 result2) -> result1
  _ -> (Error "First")
-- Evaluate Rest
eval env (Rest a) = case ((eval env a)) of
  (Error x) -> Error x
  (Pair result1 result2) -> result2
  _ -> (Error "Rest")
-- Evaluate variable lookup
eval env (Var name) = case (Data.Map.lookup name env) of
  Just a -> a -- 'a' is of type Value; return the found value
  Nothing -> Error ("Var") -- Variable not found in the environment; return an error

-- Catch-all for other expressions
eval env (If cond thenBranch elseBranch) =
  case eval env cond of
    Error msg -> Error msg -- propagate errors
    F -> eval env elseBranch -- if condition is false, evaluate else branch
    _ -> eval env thenBranch -- if condition is true (or any non-false value), evaluate then branch
eval env (Lambda args body)
  | length args /= length (unique args) = Error "Lambda"
  | otherwise = Closure args env body
eval env (App fnExpr argExprs) =
  case eval env fnExpr of
    Error msg -> Error msg
    Closure params cenv body ->
      if length params /= length argExprs
        then Error "App"
        else
          let argExpr_result = map (eval env) argExprs
              error_lst = filter isError argExpr_result
           in if null error_lst
                then
                  let newEnv = foldl (\env (param, argExpr) -> Data.Map.insert param argExpr env) cenv (zip params argExpr_result)
                   in eval newEnv body
                else head error_lst

isError :: Value -> Bool
isError (Error _) = True
isError _ = False

-- | Helper function to obtain a list of unique elements in a list
-- Example:
--   ghci> unique [1, 2, 3, 4]
--   [1,2,3,4]
--   ghci> unique [1, 2, 3, 4, 4]
--   [1,2,3,4]
unique :: (Eq a) => [a] -> [a]
unique [] = []
unique (x : xs)
  | elem x xs = unique xs -- If x appears later in the list, skip it
  | otherwise = x : unique xs -- Otherwise, keep x
  -- Note: this implementation keeps the last occurrence of each element

racketifyValue :: Value -> String
racketifyValue T = "#t"
racketifyValue F = "#f"
racketifyValue (Num x) = show x
racketifyValue Empty = "'()"
racketifyValue (Pair a b) = "(cons " ++ racketifyValue a ++ " " ++ racketifyValue b ++ ")"
racketifyValue (Closure _ _ _) = error "can't racketify a closure"
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

-- Comments --
-- I work in a group with another guy. 
-- My partner is Saabit Zubairi, he made a piazza here: https://piazza.com/class/mf2crsst8ou5h5/post/77