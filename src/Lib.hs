module Lib
    (
      dotProd,
      sigmoid,
      sigmoid',
      layer,
      -- sequential,
      errorCalc,
      mserror
    ) where

import Data.List (genericLength)
import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel (mapMatrixWithIndex, mapVectorWithIndex)

expectedOutput:: (Floating a, Element a) => Vector a
expectedOutput = fromList [0, 1, 1, 0]

startingBiases :: (Floating a, Element a) => Vector a
startingBiases = fromList $ replicate 4 (-0.5)


lr:: (Floating a, Element a) => a
lr = 0.001

startingWeights :: (Floating a, Element a) => Matrix a
startingWeights = (4><2) $ replicate 8 0.25

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
layer a' w' b' g = fromList $ map (\(x, idx) -> g $ dotProd (inputs!!idx) (weights!!idx) + (biases!!idx)) $ withIndex $ toList $ head (toColumns a')
  where
    inputs = map toList $ toRows a'
    weights = map toList $ toRows w'
    biases = toList b' 

errorCalc xs' ys' = [x - y | x <- xs', y <- ys']

-- sequential :: (Element a, Floating a) => Matrix a -> Matrix a -> Vector a -> Matrix a -> Vector a -> (a -> a) -> Vector a
-- sequential x w1 b1 w2 b2 g = layer ( layer x w1 b1 g ) w2 b2 g
--
getParameters :: (Floating a, Element a, Ord a) => Int -> Matrix a -> Matrix a -> Vector a -> (Matrix a, Vector a)
getParameters nLayers input weights biases = getParametersHelper nLayers input startingWeights startingBiases
  where 
    getParametersHelper :: (Floating a, Element a, Ord a) => Int -> Matrix a -> Matrix a -> Vector a -> (Matrix a, Vector a)
    getParametersHelper 0 o w b = (w, b)--fromList [if x < 0.5 then 0 else 1 | x <- toList v]
      where v = flatten o

    getParametersHelper n a w b = getParametersHelper (n - 1) (fromLists [toList res]) mappedShit mappedVec
      where 
        res = layer a w b sigmoid
        error = errorCalc (toList expectedOutput) $ toList res 
        adjustment = [x - y |x <- map sigmoid' (toList res), y <- error]
        mappedShit = mapMatrixWithIndex (\(x, y) mxelem -> mxelem - lr * adjustment!!x) w
        mappedVec = mapVectorWithIndex (\idx item -> item - lr * adjustment!!idx) b
    startingWeights = (4><2) $ replicate 8 0.0

final :: (Floating a, Element a, Ord a) => Int -> Matrix a -> Matrix a -> Vector a -> (Matrix a, Vector a)
final 0 _ w b = (w, b)
final n a w b = uncurry (final (n - 1) a) newParams
  where
    newParams = getParameters n a w b

mserror :: (Floating a, Element a) => Vector a -> Vector a -> [a]
mserror xs ys = {-(1 / genericLength lx) * -}mserrorHelper lx ly 
  where
    mserrorHelper xs' ys' = map (\(x, idx) -> x - ys'!!idx) $ withIndex $ xs'
    lx = toList xs
    ly = toList ys

