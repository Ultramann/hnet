{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}

module Neural.Matrix (
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

newtype Activation = Activation { activate :: Vector R -> Vector R }
data FullyConnected = FullyConnected { biases  :: Vector R
                                     , weights :: Weights R }
type Network = Networkable a => [a]

class Networkable f where
  feedForward :: f -> Vector R -> Vector R

instance Networkable Activation where
  feedForward a = activate

instance Networkable FullyConnected where
  feedForward FullyConnected {..} = (biases +) . (weights #>)

--parameters :: Layer -> (Biases, Weights)
--parameters = biases &&& weights

initNet :: R -> [Int] -> [Activation] -> IO Network
initNet b szs@(_:lns) as = concat $ transpose [fcs, as]
  where fcs = zipWith FullyConnected (vector . flip replicate b <$> lns) (zipWithM randn szs lns)

relu :: Activation
relu = cmap $ max 0

relu' :: Activation
relu' = cmap f
  where f x | x < 0     = 0
            | otherwise = 1

categoricalCrossEntropy :: RealFloat a => [a] -> [a] -> a
categoricalCrossEntropy ys = negate . sum . zipWith (*) ys . map log . softmax

softmax :: RealFloat a => [a] -> [a]
softmax ts = map (/ sum exps) exps
  where exps = exp <$> ts

feedLayer :: (Vector R, Vector R) -> Layer -> (Vector R, Vector R)
feedLayer (_, an) (Layer aF bs ws) = (id &&& fromList . aF . toList) $ bs + an <# ws

forwardPass :: Vector R -> [Layer] -> [(Vector R, Vector R)]
forwardPass = scanl' feedLayer . (id &&& id)

propagate :: (Vector R, Matrix R) -> Vector R -> Vector R
propagate (zn, wn) dldn = wn #> dldn * dadz
  where dadz = fromList . relu' . toList $ zn

backPropagate :: Vector R -> [Vector R] -> [Matrix R] -> [Vector R]
backPropagate dldL tls lws = tail $ scanr propagate dldL $ zip tls lws

updateLayer :: R -> (Vector R, Vector R, Layer) -> Layer
updateLayer s (a, e, (Layer aF bs ws)) = Layer aF newBs newWs
  where newWs = ws - scalar s * outer a e
        newBs = bs - scalar s * e

updateNet :: (forall a. Reifies a Tape => [Reverse a R]
                                       -> [Reverse a R]
                                       -> Reverse a R) ->
             R -> Vector R -> Vector R -> Network -> Network
updateNet c s xs ys ls = map (updateLayer s) $ zip3 (init ans) dls ls
  where fps  = forwardPass xs ls
        ans  = snd <$> fps
        yHat = auto <$> (toList . last) ans
        dldL = fromList $ grad (c (auto <$> (toList ys))) yHat
        dls  = backPropagate dldL (fst <$> init fps) (weights <$> ls)
