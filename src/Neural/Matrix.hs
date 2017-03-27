{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}

module Neural.Matrix(
    initNet, updateNet, feedLayer, forwardPass, chain, backProp,
    relu, softmax, categoricalCrossEntropy
) where

import Control.Monad (zipWithM)
import Data.List (foldl', scanl', transpose)
import Control.Arrow ((&&&))
import Numeric.AD (grad)
import Numeric.AD.Mode (auto)
import Numeric.AD.Internal.Reverse (Reverse, Tape)
import Data.Reflection (Reifies)
import Numeric.LinearAlgebra

type Biases = Vector R
type Weights = Matrix R
type Activation = [R] -> [R]
type Layer = (Activation, Biases, Weights)

parameters :: Layer -> (Biases, Weights)
parameters (_, bs, ws) = (bs, ws)

weights :: Layer -> Weights
weights (_, _, ws) = ws

initNet :: R -> [Int] -> [Activation] -> IO [Layer]
initNet b szs@(_:lns) acts = layer <$> zipWithM randn szs lns
  where layer = zip3 acts $ vector . flip replicate b <$> lns

relu :: Activation
relu = map $ max 0

relu' :: Activation
relu' = map f
  where f x | x < 0     = 0
            | otherwise = 1

reluCost :: RealFloat a => [a] -> [a] -> a
reluCost ys = (*0.5) . sum . map (^2) . zipWith (-) ys . relu
  where relu = map $ max 0

categoricalCrossEntropy :: RealFloat a => [a] -> [a] -> a
categoricalCrossEntropy ys = negate . sum . zipWith (*) ys . map log . softmax

softmax :: RealFloat a => [a] -> [a]
softmax ts = let exps = exp <$> ts in map (/ sum exps) exps

feedLayer :: (Vector R, Vector R) -> Layer -> (Vector R, Vector R)
feedLayer (_, ins) (act, bs, ws) =  (id &&& fromList . act . toList) $ bs + ins <# ws

forwardPass :: Vector R -> [Layer] -> [(Vector R, Vector R)]
forwardPass = scanl' feedLayer . (id &&& id)

chain :: (Vector R, Matrix R) -> Vector R -> Vector R
chain (transLayer, layerWeights) err = chainError * dTransLayer
  where chainError  = err <# tr layerWeights
        dTransLayer = fromList . relu' . toList $ transLayer

backProp :: Vector R -> [Vector R] -> [Matrix R] -> [Vector R]
backProp e tls lws = tail $ scanr chain e $ zip tls lws

updateLayer :: R -> (Vector R, Vector R, Layer) -> Layer
updateLayer s (a, e, (actF, bs, ws)) = (actF, newBiases, newWeights)
  where newWeights = ws - scalar s * outer a e
        newBiases  = bs - scalar s * e

updateNet :: (forall a. Reifies a Tape => [Reverse a R]
                                       -> [Reverse a R]
                                       -> Reverse a R) ->
             R -> Vector R -> Vector R -> [Layer] -> [Layer]
updateNet c s xs ys ls = map (updateLayer s) $ zip3 (init as) es ls
  where fps = forwardPass xs ls
        as  = snd <$> fps
        ps  = auto <$> (toList . last) as
        e1  = fromList $ grad (c (auto <$> (toList ys))) ps
        es  = backProp e1 (fst <$> init fps) (weights <$> ls)
