module Main (main) where

import Neural.Matrix
import Data.List (scanl, foldl')
import Numeric.LinearAlgebra (toList, vector)
import Control.Monad (forM_)
import Codec.Compression.GZip (decompress)
import qualified Data.ByteString.Lazy as BS

getImage s n = fromIntegral . BS.index s . (n*28^2 + 16 +) <$> [0..28^2 - 1]
getX     s n = (vector (getImage s n)) / 256
getLabel s n = fromIntegral $ BS.index s (n + 8)
getY     s n = vector $ fromIntegral . fromEnum . (getLabel s n ==) <$> [0..9]

printImage imgs = putStrLn . unlines . take 28 . map (take 28) . iterate (drop 28)
                           . map pixel2Char . getImage imgs
  where pixel2Char n = let s = " ·:o◍O0@" in s !! (fromIntegral n * length s `div` 256)

display d prob = show d ++ ": " ++ show prob

small = [[0..99], [100..299], [300..599], [600..999]]
medium = [[0..299], [300..999], [1000..2199], [2200..4599]]
large = [[0..999], [1000..2999], [3000..5999], [6000..9999]]
rapid = [[0..299], [300..599], [600..899], [900..1199], [1200..1499], [1500..1799], [1800..2099]]

main :: IO ()
main = do
  [trainI, trainL, testI, testL] <- mapM ((decompress  <$>) . BS.readFile . ("examples/mnistData/"++))
                                    ["train-images-idx3-ubyte.gz"
                                    ,"train-labels-idx1-ubyte.gz"
                                    ,"t10k-images-idx3-ubyte.gz"
                                    ,"t10k-labels-idx1-ubyte.gz"]
  net <- initNet 1 [784, 30, 10] [relu, relu]
  let n = 42
  printImage testI n

  let
    example = getX testI n
    randomUpdate net x = updateNet categoricalCrossEntropy 0.002 (getX trainI x) (getY trainL x) net
    trainingNet = scanl (foldl' randomUpdate) net large
    trainedNet = last trainingNet
  forM_ trainingNet $ putStrLn . unlines . zipWith display [0..9] . softmax
                               . toList . last . snd . forwardPass example
