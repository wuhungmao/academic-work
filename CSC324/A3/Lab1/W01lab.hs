-- This lists what this module exports. Don't change this!

-- | * CSC324H5 Fall 2024: Week 1 Lab *
-- Module:        W01lab
-- Description:   Week 1 Lab: Introduction to Racket and Haskell
-- Copyright: (c) University of Toronto Mississsauga
--               CSC324 Principles of Programming Languages, Fall 2024
--
-- In this part of the exercise, you'll get started writing some simple
-- functions in Racket. Since this is likely your first time using Racket,
-- we strongly recommend going through some of the documentation we listed
-- under the "Software" page as you work through this exercise.
-- In comments below, we also give some links to documentation to built-in
-- functions for standard data types (numbers, strings, lists) that we want
-- you to become familiar with.
--
-- Finally, you'll notice the (module+ test ...) expressions interleaved with
-- the function definitions; this is a standard Racket convention for simple
-- unit tests that we'll use throughout the course. Please read them carefully,
-- and add tests of your own!
module W01lab
  ( percentage,
    nCopies,
    appears,
  )
where

import Data.Text.Lazy.Builder.Int (decimal)
import Test.QuickCheck (Property, quickCheck, (==>))

-------------------------------------------------------------------------------

-- * Note about type signatures

--
-- Unlike Racket, Haskell is /statically-typed/. We'll go into more detail about
-- what this means later in the course, but for now we've provided type signatures
-- for the functions here to simplify any compiler error messages you might
-- receive. (Don't change them; they're required to compile against our tests.)
-------------------------------------------------------------------------------

-- | Convert the decimal value into a percentage, rounded
--   to the nearest integer.
-- __Note__: use the @round@ function to convert from floating-point types
-- to @Int@.
percentage :: Float -> Int
percentage decimal =
  -- TODO: replace `undefined` with a proper function body.
  round (decimal * 100)

-- | The simplest "property-based test" is simply a unit test; note the type.
prop_percentage0 :: Bool
prop_percentage0 = percentage 0.2023 == 20

prop_percentage1 :: Bool
prop_percentage1 = percentage 1.25555 == 126

-------------------------------------------------------------------------------

-- * Recursion with numbers

--
-- For the recursive functions, we recommend doing these in two ways:
--
--   1. First, write them using @if@ expressions, as you would in Racket.
--   2. Then when that works, use /pattern-matching/ to simplify the definitions
--      (<http://learnyouahaskell.com/syntax-in-functions#pattern-matching>).
--
-- Remember: Strings are simply lists of characters. (@String === [Char]@)
-- Read more about manipulating lists at
-- <http://learnyouahaskell.com/starting-out#an-intro-to-lists>.

-- | Returns a new string that contains @n@ copies of the input string.
nCopies :: String -> Int -> String
nCopies s n = concat (replicate n s)

-- | This is a QuickCheck property that says,
-- "If n >= 0, then when you call nCopies on a string s and int n,
-- the length of the resulting string is equal to
-- n * the length of the original string."
--
-- QuickCheck verifies this property holds for a random selection of
-- inputs (by default, choosing 100 different inputs).
prop_nCopiesLength :: String -> Int -> Property
prop_nCopiesLength s n = n >= 0 ==> length (nCopies s n) == (length s * n)

-------------------------------------------------------------------------------

-- * Recursion with lists

-------------------------------------------------------------------------------

-- | Returns whether a string appears in a list of strings
--
-- We've given you a recursive template here to start from.
-- But noted as above, you can later try simplifying this definition
-- using pattern matching.
appears :: String -> [String] -> Bool
appears s lst =
  if null lst
    then False
    else
      let firstVal = head lst
          stringAppearInRest = tail lst
       in if s == firstVal
            then True
            else appears s stringAppearInRest

-- | This is a QuickCheck property that says,
-- "When you call appears on the string s and the list [s, s]
--  you should get True"
--
-- QuickCheck verifies this property holds for a random selection of
-- inputs (by default, choosing 100 different inputs).
prop_nAppearsTwice :: String -> Bool
prop_nAppearsTwice s = appears s [s, s]

-------------------------------------------------------------------------------

-- * Main function (for testing purposes only)

-------------------------------------------------------------------------------
prop_percentage2 :: Bool
prop_percentage2 = percentage 0.871 == 87

prop_nCopiesLength1 :: Bool
prop_nCopiesLength1 = nCopies "CSC324! " 3 == "CSC324! " ++ "CSC324! " ++ "CSC324! "

prop_nAppearsTwice1 :: Bool
prop_nAppearsTwice1 = appears "a" ["b", "c", "d"] == False

-- This main function is executed when you compile and run this Haskell file.
-- It runs the QuickCheck tests; we'll talk about "do" notation much later in
-- the course, but for now if you want to add your own tests, just define them
-- above, and add a new `quickCheck` line below.
main :: IO ()
main = do
  quickCheck prop_percentage0
  quickCheck prop_percentage1
  quickCheck prop_nCopiesLength
  quickCheck prop_nCopiesLength1
  quickCheck prop_nAppearsTwice
  quickCheck prop_nAppearsTwice1
