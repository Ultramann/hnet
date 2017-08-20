{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}

module Neural.Matrix (
    initNet, updateNet, forwardPass,
    backPropagate, relu, softmax, categoricalCrossEntropy
) where

import Control.Monad (zipWithM)
import Data.List (scanl')
import Control.Arrow ((&&&))
import Numeric.AD (grad)
import Numeric.AD.Mode (auto)
import Numeric.AD.Internal.Reverse (Reverse, Tape)
import Data.Reflection (Reifies)
import Numeric.LinearAlgebra

type Gradient = Vector R

class Differentiable d where
  feedForward  :: d -> Vector R -> Vector R
  passBackward :: d -> Gradient -> Gradient

data Activation = Activation { activate :: Vector R -> Vector R 
                             , derive   :: Vector R -> Vector R }

instance Differentiable Activation where
  feedForward  = activate
  passBackward = derive

data Dense = Dense { biases  :: Vector R
                   , weights :: Matrix R }

instance Differentiable Dense where
  feedForward  Dense {..} = (biases +) . (<# weights)
  passBackward Dense {..} = (weights #>)

data Network = Network { denses      :: [Dense]
                       , activations :: [Activation] }

initNet :: R -> [Int] -> [Activation] -> IO Network
initNet b szs@(_:lns) as = do 
      ws <- zipWithM randn szs lns 
      let bs = vector . flip replicate b <$> lns
          ds = zipWith Dense bs ws
      return $ Network { denses = ds, activations = as }

relu :: Activation
relu = Activation { activate = cmap $ max 0
                  , derive = let f x = if x < 0 then 0 else 1 in cmap f }

categoricalCrossEntropy :: RealFloat a => [a] -> [a] -> a
categoricalCrossEntropy ys = negate . sum . zipWith (*) ys . map log . softmax

softmax :: RealFloat a => [a] -> [a]
softmax ts = let exps = exp <$> ts in map (/ sum exps) exps

forwardPass :: Vector R -> Network -> [Vector R]
forwardPass xs Network {..} = scanl' feedLayer xs $ zip denses activations
  where feedLayer t (d, a) = feedForward a $ feedForward d t

backPropagate :: Gradient -> Network -> [Vector R]
backPropagate dldL Network {..} = scanr propThruLayer dldL $ zip denses activations
  where propThruLayer (d, a) dldz = passBackward d $ passBackward a dldz

update :: R -> (Dense, (Gradient, Vector R)) -> Dense
update s (Dense {..}, (dldn, anp)) = Dense newBiases newWeights 
  where newBiases = biases - scalar s * dldn
        newWeights = weights - scalar s * outer anp dldn

updateNet :: (forall a. Reifies a Tape => [Reverse a R]
                                       -> [Reverse a R]
                                       -> Reverse a R) ->
             R -> Vector R -> Vector R -> Network -> Network
updateNet c s xs ys net = net { denses = update s <$> zip (denses net) (zip dldns as) }
  where (as,a) = (init &&& last) $ forwardPass xs net
        yHat   = map auto . toList $ a
        dldL   = fromList $ grad (c (auto <$> (toList ys))) yHat
        dldns  = backPropagate dldL net
