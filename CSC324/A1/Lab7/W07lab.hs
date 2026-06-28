{-|
Module:        W07lab
Description:   Week 7 Lab
Copyright: (c) University of Toronto Mississsauga
               CSC324 Principles of Programming Languages, Fall 2023
-}

-- This lists what this module exports. Don't change this!
module W07lab
  (
    mapMaybes,
    composeMaybe,
    foldMaybe, 
    applyBinaryMaybe,
    collectMaybes,
    Person(..),
    Robot(..),
    Organization(..),
    robotCompany
  )
where

-------------------------------------------------------------------------------
-- * Task 1: Practice with maybe
-------------------------------------------------------------------------------

mapMaybes :: (a -> b) -> [Maybe a] -> [Maybe b]
mapMaybes f maybe_a_lst = map (\maybe_a -> case maybe_a of
                                        Nothing -> Nothing
                                        Just a  -> Just (f a)
                                ) maybe_a_lst

composeMaybe :: (a -> Maybe b) -> (b -> Maybe c) -> (a -> Maybe c)
composeMaybe f g = \a -> case f a of
                          Nothing -> Nothing
                          Just b  -> g b

foldMaybe :: (b -> a -> Maybe b) -> b -> [a] -> Maybe b
foldMaybe fold_func accu_b elem_lst = foldl (\maybe_b elem -> case maybe_b of
                                                                      Nothing -> Nothing  
                                                                      Just b  -> fold_func b elem
                                                  ) (Just accu_b) elem_lst

applyBinaryMaybe :: (a -> b -> c) -> Maybe a -> Maybe b -> Maybe c
applyBinaryMaybe f maybe_a maybe_b =  case maybe_a of 
                                        Nothing -> Nothing
                                        Just a  -> case maybe_b of
                                                      Nothing -> Nothing
                                                      Just b  -> Just (f a b)

collectMaybes :: [Maybe a] -> Maybe [a]
collectMaybes (maybe_a_lst) = foldr step (Just []) maybe_a_lst
  where
    step :: Maybe a -> Maybe [a] -> Maybe [a]
    step maybe_a maybe_acc = applyBinaryMaybe (:) maybe_a maybe_acc

-------------------------------------------------------------------------------
-- * Task 2: Practice with Functor
-------------------------------------------------------------------------------

-- Here, we'll need to make Organization an instance of the Eq
-- *and* Functor typeclass.
data Person  = Person String Float      -- name, salary
             deriving (Show, Eq)
data Robot   = Robot Int                -- identifier
             deriving (Show, Eq)
data Organization p = Individual p             -- organization of one
                    | Team p [Organization p]  -- team leader, and list of sub-orgs
                    deriving (Show, Eq)

instance Functor Organization where
    fmap func (Individual p) = Individual (func p)
    fmap func (Team p org_lst) = Team (func p) (map (fmap func) org_lst)

-- robot organization:
robot1   = Robot 1
robotOrg = Individual robot1

-- example:
owner   = Person "Janet" 100000
cto     = Person "Larry"  90000
cfo     = Person "Mike"   90000
intern  = Person "Sam"    40000
company = Team owner [Team cto [Individual intern],
                      Individual cfo]

-- Use a call to `fmap` to turn the example value `company`
-- into an organization with the same structure, but populated
-- entirely by robots. 
robotize :: Person -> Robot
robotize p = Robot 0

robotCompany = fmap robotize company


