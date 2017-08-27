{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE RecordWildCards #-}

module Neural.Matrix (
    Dense(..), Network(..), initNet, updateNet, feedForward, forwardPass,
    passBackward, backPropagate, relu, softmax, categoricalCrossEntropy
) where

import Control.Monad (zipWithM)
import Data.List (foldl', scanl')
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

data Dense = Dense { weights :: Matrix R
                   , biases  :: Vector R }

instance Differentiable Dense where
  feedForward  Dense {..} = (biases +) . (<# weights)
  passBackward Dense {..} = (weights #>)

data Network = Network { denses      :: [Dense]
                       , activations :: [Activation] }

initNet :: R -> [Int] -> [Activation] -> IO Network
initNet b szs@(_:lns) as = do
      ws <- zipWithM randn szs lns
      let bs = vector . flip replicate b <$> lns
          ds = zipWith Dense ws bs
      return $ Network { denses = ds, activations = as }

relu :: Activation
relu = Activation { activate = cmap $ max 0
                  , derive   = let f x = if x <= 0 then 0 else 1 in cmap f }

categoricalCrossEntropy :: RealFloat a => [a] -> [a] -> a
categoricalCrossEntropy ys = negate . sum . zipWith (*) ys . map log . softmax

softmax :: RealFloat a => [a] -> [a]
softmax ts = let exps = exp <$> ts in map (/ sum exps) exps

forwardPass :: Vector R -> Network -> ([Vector R], [Vector R])
forwardPass xs Network {..} = separate . scanl' feedLayer (xs, xs) $ zip denses activations
  where feedLayer (_, al) (dn, af) = let z = feedForward dn al in (z, feedForward af z)
        separate = foldr (\(z,a) (zs, as) -> (z:zs, a:as)) ([], [])

backPropagate :: Gradient -> Network -> [Vector R] -> [Vector R]
backPropagate dldL Network {..} = scanr propThruLayer dldL . zip3 denses activations
  where propThruLayer (dn, af, z) dldz = passBackward dn $ dldz * passBackward af z

update :: R -> (Gradient, Vector R, Dense) -> Dense
update s (dldn, anp, Dense {..}) = Dense newWeights newBiases
  where newBiases = biases - scalar s * dldn
        newWeights = weights - scalar s * outer anp dldn

updateNet :: (forall a. Reifies a Tape => [Reverse a R]
                                       -> [Reverse a R]
                                       -> Reverse a R) ->
             R -> Vector R -> Vector R -> Network -> Network
updateNet c s xs ys net = net { denses = newDenses }
  where (zs, as)  = forwardPass xs net
        yHat      = map auto . toList . last $ as
        dldL      = fromList $ grad (c (auto <$> (toList ys))) yHat
        dldns     = tail $ backPropagate dldL net $ tail zs
        newDenses = update s <$> zip3 dldns (init as) (denses net)
