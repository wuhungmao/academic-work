-- This lists what this module exports. Don't change this!

-- | * CSC324H5 Fall 2024: Week 2 Lab *
-- Module:        W02lab
-- Description:   Week 2 Lab: Pattern Matching and Recursion
-- Copyright: (c) University of Toronto Mississsauga
--               CSC324 Principles of Programming Languages, Fall 2024
module W02lab
  ( replaceAll,
    replaceAllHelper,
  )
where

import Test.QuickCheck (Property, quickCheck, (==>))

-------------------------------------------------------------------------------

-- * Note about type signatures

--
-- Unlike Racket, Haskell is /statically-typed/. We'll go into more detail about
-- what this means later in the course, but for now we've provided type signatures
-- for the functions here to simplify any compiler error messages you might
-- receive. (Don't change them; they're required to compile against our tests.)
-------------------------------------------------------------------------------

-- | Replace every instance of a number in a list with a second number
replaceAll :: [Int] -> Int -> Int -> [Int]
replaceAll lst id with = replaceAllHelper lst id with []

-- | Helper function for replaceAll
replaceAllHelper :: [Int] -> Int -> Int -> [Int] -> [Int]
replaceAllHelper [] _ _ acc = reverse acc
replaceAllHelper (x : xs) id with acc =
  if x == id
    then replaceAllHelper xs id with (with : acc)
    else replaceAllHelper xs id with (x : acc)

-- | The simplest "property-based test" is simply a unit test; note the type.
prop_replaceAllSingle :: Bool
prop_replaceAllSingle = (replaceAll [0, 1, 2, 3] 0 1) == [1, 1, 2, 3]

prop_replaceAll :: [Int] -> Bool
prop_replaceAll lst = not (elem 1 (replaceAll lst 1 2))

-- You may wish write additional simple unit tests like prop_replaceAllSingle
-- Add the tests you wrote to the end of this file, in the `main` block
-- to actually run those tests.

-------------------------------------------------------------------------------

-- * Main function (for testing purposes only)

-------------------------------------------------------------------------------

-- This main function is executed when you compile and run this Haskell file.
-- It runs the QuickCheck tests; we'll talk about "do" notation much later in
-- the course, but for now if you want to add your own tests, just define them
-- above, and add a new `quickCheck` line below.
main :: IO ()
main = do
  quickCheck prop_replaceAllSingle
  quickCheck prop_replaceAll
