{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}

module Neural.Matrix(
    initNet, updateNet, feedLayer, forwardPass, propagate,
    backPropagate, relu, softmax, categoricalCrossEntropy
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

categoricalCrossEntropy :: RealFloat a => [a] -> [a] -> a
categoricalCrossEntropy ys = negate . sum . zipWith (*) ys . map log . softmax

softmax :: RealFloat a => [a] -> [a]
softmax ts = let exps = exp <$> ts in map (/ sum exps) exps

feedLayer :: (Vector R, Vector R) -> Layer -> (Vector R, Vector R)
feedLayer (_, an) (aF, bs, ws) = (id &&& fromList . aF . toList) $ bs + an <# ws

forwardPass :: Vector R -> [Layer] -> [(Vector R, Vector R)]
forwardPass = scanl' feedLayer . (id &&& id)

propagate :: (Vector R, Matrix R) -> Vector R -> Vector R
propagate (zn, wn) dldn = wn #> dldn * dadz
  where dadz = fromList . relu' . toList $ zn

backPropagate :: Vector R -> [Vector R] -> [Matrix R] -> [Vector R]
backPropagate dldL tls lws = tail $ scanr propagate dldL $ zip tls lws

updateLayer :: R -> (Vector R, Vector R, Layer) -> Layer
updateLayer s (a, e, (aF, bs, ws)) = (aF, newBs, newWs)
  where newWs = ws - scalar s * outer a e
        newBs = bs - scalar s * e

updateNet :: (forall a. Reifies a Tape => [Reverse a R]
                                       -> [Reverse a R]
                                       -> Reverse a R) ->
             R -> Vector R -> Vector R -> [Layer] -> [Layer]
updateNet c s xs ys ls = map (updateLayer s) $ zip3 (init ans) dls ls
  where fps  = forwardPass xs ls
        ans  = snd <$> fps
        yhat = auto <$> (toList . last) ans
        dldL = fromList $ grad (c (auto <$> (toList ys))) yhat
        dls  = backPropagate dldL (fst <$> init fps) (weights <$> ls)
