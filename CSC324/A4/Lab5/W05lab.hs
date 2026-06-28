{-|  * CSC324H5 Fall 2025: Week 5 Lab * 
Module:        W05lab
Description:   Week 5 Lab: Type Systems
Copyright: (c) University of Toronto Mississsauga
               CSC324 Principles of Programming Languages, Fall 2025

-}

-- This lists what this module exports. Don't change this!
module W05lab
  (
    postsWithMinimumComments,
    removePhoneNumbers,
    retrieveAllCommentedPosts,
    followingBack,
    Account(..),
    Post(..),
  )
where

-- Remember that you may not add any additional imports
import Test.QuickCheck (Property, quickCheck, (==>))

data Account = Account String String (Maybe String) [Account] [Post] deriving (Show, Eq)   -- username, email, phone number (if provided), accounts that this user is following, posts
data Post = Post Int (Either String Int) deriving (Show, Eq)                               -- postidentifier, number of comments on the post (either Left "Disabled" or Right (number of comments))

-------------------------------------------------------------------------------
-- * Note on Either Type

-- In class, you may have learned about the Maybe type constructor! In this lab, we will be working with 
-- the Either type constructor as well (as seen in the Post type). You can read more about how the Either
-- type constructor works in this link. 

-- https://hackage.haskell.org/package/base-4.21.0.0/docs/Data-Either.html

-- Step 1: Write the function postsWithMinimumComments that takes a list of posts, a minimum comment count, and
-- returns posts within that list that have comments enabled and have more comments than the minimum comment
-- count.
helper1 :: Int -> Post -> Bool
helper1 minComments (Post id comments_or_disabled) = 
  case comments_or_disabled of
    Left _ -> False
    Right comment -> if comment > minComments
                    then True
                    else False

-- | Return all posts with comments enabled and has more comments than the argument provided
postsWithMinimumComments :: [Post] -> Int -> [Post]
postsWithMinimumComments post_lst minComments = 
  filter (helper1 minComments) post_lst

helper2 :: Account -> Account
helper2 (Account username email maybe_phone_number accounts posts) = 
  case maybe_phone_number of 
    Just phone_number -> Account username email Nothing accounts posts
    Nothing -> Account username email Nothing accounts posts


-- | Remove all phone numbers for accounts that have one provided
removePhoneNumbers :: [Account] -> [Account]
removePhoneNumbers account = 
  map helper2 account

-- Step 3: Write the function followingBack that takes an account A, a list of accounts, and returns a list of accounts
-- that are following account A back. It might be useful to know that since we are deriving Eq in both of our types, you
-- are able to compare Account instances directly using (==)

helper3 :: Account -> Account -> Bool
helper3 targetAccount (Account _ _ _ followedAccounts _) = 
  -- 'any' takes a predicate (\follower -> follower == targetAccount) 
  -- and checks if it's True for AT LEAST ONE account in followedAccounts.
  any (\follower -> follower == targetAccount) followedAccounts

-- | Given an account and a list of accounts, return the list of accounts that are following this account back
followingBack :: Account -> [Account] -> [Account]
followingBack account account_lst = 
  filter (helper3 account) account_lst


helper4 :: [Account] -> [Post] -> [Post]
helper4 [] post_lst = post_lst 
helper4 (account : accounts_lst) post_lst = 
  case account of
    (Account _ _ _ _ posts) -> 
      let accu_lst = posts ++ post_lst
      in helper4 accounts_lst accu_lst

-- | Go through all accounts and return all posts (with commenting enabled) across all accounts, in one list 
retrieveAllCommentedPosts :: [Account] -> [Post]
retrieveAllCommentedPosts accounts_lst = 
  helper4 accounts_lst []


-------------------------------------------------------------------------------
-- * Main function (for testing purposes only)
------------------------------------------------------------------------------- 

prop_postsWithMinimumComments :: Bool
prop_postsWithMinimumComments = [Post 1 (Right 124)] == postsWithMinimumComments [Post 1 (Right 124), Post 2 (Left "Disabled")] 123

-- This main function is executed when you compile and run this Haskell file.
-- It runs the QuickCheck tests; we'll talk about "do" notation much later in
-- the course, but for now if you want to add your own tests, just define them
-- above, and add a new `quickCheck` line below.
main :: IO ()
main = do
  quickCheck prop_postsWithMinimumComments
    -- TODO: add other tests


