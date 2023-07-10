module Lib
    (
      dotProd,
      sigmoid,
      sigmoid',
      layer,
      -- sequential,
      mserror
    ) where

import Data.List (genericLength)
import Numeric.LinearAlgebra

expectedOutput:: (Num a, Element a) => Vector a
expectedOutput = fromList [0, 1, 1, 0]

startingBiases :: (Num a, Element a) => Vector a
startingBiases = fromList $ replicate 4 0

epochs = 1
lr = 0.1 :: Double

inputData = (4><2)
  [ 0, 0
  , 0, 1
  , 1, 0
  , 1, 1
  ] :: Matrix R

withIndex :: [a] -> [(a, Int)]
withIndex x = zip x [0..]

dotProd :: (Num a) => [a] -> [a] -> a
dotProd xs ys = sum $ [x * y | x <- xs, y <- ys]

sigmoid :: (Floating a) => a -> a
sigmoid x = 1 / (1 + exp(-x))

sigmoid' :: (Floating a) => a -> a 
sigmoid' x = s * (1 - s)
  where s = sigmoid x

layer :: (Element a, Floating a) => Matrix a -> Matrix a -> Vector a -> (a -> a) -> Vector a
layer a' w' b' g = fromList $ map (\(x, idx) -> g $ dotProd (inputs!!idx) (weights!!idx) + x) $ withIndex biases
  where
    inputs = map toList $ toRows a'
    weights = map toList $ toRows w'
    biases = toList b' 

-- sequential :: (Element a, Floating a) => Matrix a -> Matrix a -> Vector a -> Matrix a -> Vector a -> (a -> a) -> Vector a
-- sequential x w1 b1 w2 b2 g = layer ( layer x w1 b1 g ) w2 b2 g
--
{-getParameters :: (Floating a, Element a, Ord a) => Matrix a -> Vector a
getParameters input = getParametersHelper epochs input startingWeights startingBiases
  where 
    getParametersHelper :: (Floating a, Element a, Ord a) => Int -> Matrix a -> Matrix a -> Vector a -> Vector a
    getParametersHelper 0 o w b = fromList [if x < 0.5 then 0 else 1 | x <- toList v]
      where v = flatten o
    getParametersHelper e a w b = getParametersHelper (e - 1) (layer a w b sigmoid) w b
    startingWeights = (4><2) $ replicate 8 0
-}

mserror :: (Floating a, Element a) => Vector a -> Vector a -> a 
mserror xs ys = (1 / genericLength lx) * mserrorHelper lx ly 
  where
    mserrorHelper xs' ys' = sum $ [(x - y) ^ 2 | x <- xs', y <- ys' ]
    lx = toList xs
    ly = toList ys
