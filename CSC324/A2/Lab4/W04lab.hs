-- This lists what this module exports. Don't change this!

-- |  * CSC324H5 Fall 2024: Week 4 Lab *
-- Module:        W04lab
-- Description:   Week 4 Lab: Functional Data Structures
-- Copyright: (c) University of Toronto Mississsauga
--               CSC324 Principles of Programming Languages, Fall 2024
module W04lab
  ( alookup,
    aset,
    adel,
  )
where

-- Remember that you may not add any additional imports
import Test.QuickCheck (Property, quickCheck, (==>))

-- This is a **type alias** to make the type signatures of our
-- functions more easily readable.
-- Any time we write "AssocList" in a type signature, it is
-- identical to writing "[(String, Int)]"
type AssocList = [(String, Int)]

-- | Look up a key in the Association List
alookup :: AssocList -> String -> Int
alookup assoc s =
  case lookup s assoc of
    Just value -> value -- Key found, return the actual value
    Nothing -> -1 -- Key not found, return -1

-- | Set a key in the Association List
aset :: AssocList -> String -> Int -> AssocList
aset assoc k v =
  if lookup k assoc == Nothing
    then (k, v) : assoc
    else assoc

-- | Delete a key in the Association List
adel :: AssocList -> String -> AssocList
adel assoc k =
  if lookup k assoc == Nothing
    then assoc
    else filter (\(key, v) -> key /= k) assoc

-------------------------------------------------------------------------------

-- * Main function (for testing purposes only)

-------------------------------------------------------------------------------

prop_alookup :: Bool
prop_alookup = 1 == alookup [("A", 1)] "A"

-- This main function is executed when you compile and run this Haskell file.
-- It runs the QuickCheck tests; we'll talk about "do" notation much later in
-- the course, but for now if you want to add your own tests, just define them
-- above, and add a new `quickCheck` line below.
main :: IO ()
main = do
  quickCheck prop_alookup

-- TODO: add other tests
