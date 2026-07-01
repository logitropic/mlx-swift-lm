// Copyright © 2026
//
// Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/deepseekocr

import CoreImage
@preconcurrency import CoreImage.CIFilterBuiltins
import CoreGraphics
import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - Processor

public struct DeepseekOCRProcessorConfiguration: Decodable, Sendable {
    public let candidateResolutions: [[Int]]
    public let patchSize: Int
    public let downsampleRatio: Int
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageToken: String
    public let padToken: String
    public let sftFormat: String

    enum CodingKeys: String, CodingKey {
        case candidateResolutions = "candidate_resolutions"
        case patchSize = "patch_size"
        case downsampleRatio = "downsample_ratio"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageToken = "image_token"
        case padToken = "pad_token"
        case sftFormat = "sft_format"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        candidateResolutions =
            try c.decodeIfPresent([[Int]].self, forKey: .candidateResolutions) ?? [[1024, 1024]]
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        downsampleRatio = try c.decodeIfPresent(Int.self, forKey: .downsampleRatio) ?? 4
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<image>"
        padToken = try c.decodeIfPresent(String.self, forKey: .padToken) ?? "<｜▁pad▁｜>"
        sftFormat = try c.decodeIfPresent(String.self, forKey: .sftFormat) ?? "deepseek"
    }

    var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }
}

public struct DeepseekOCRProcessor: UserInputProcessor {
    public let config: DeepseekOCRProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let baseSize = 1024
    private let cropSize = 640
    private let bosTokenId = 0
    private let padTokenId = 100001

    public init(_ config: DeepseekOCRProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private var imageTokenId: Int {
        tokenizer.encode(text: config.imageToken, addSpecialTokens: false).first ?? 128815
    }

    private func normalizePrompt(_ prompt: String, imageCount: Int) throws -> String {
        guard imageCount == 1 else { throw VLMError.singleImageAllowed }
        let count = prompt.components(separatedBy: config.imageToken).count - 1
        func withPythonTemplateTrailingSpace(_ text: String) -> String {
            text.hasSuffix(" ") ? text : text + " "
        }
        if count == 0 {
            return withPythonTemplateTrailingSpace("\(config.imageToken)\n\(prompt)")
        }
        guard count == 1 else {
            throw VLMError.processing("DeepSeekOCR expects exactly one <image> token.")
        }
        return withPythonTemplateTrailingSpace(prompt)
    }

    private func closestAspectRatio(width: Int, height: Int, imageSize: Int) -> (Int, Int) {
        let aspectRatio = Double(width) / Double(height)
        let area = width * height
        var ratios: [(Int, Int)] = []
        for n in 2 ... 9 {
            for i in 1 ... n {
                for j in 1 ... n where i * j >= 2 && i * j <= 9 {
                    ratios.append((i, j))
                }
            }
        }
        ratios.sort { $0.0 * $0.1 < $1.0 * $1.1 }

        var best = (1, 1)
        var bestDiff = Double.infinity
        for ratio in ratios {
            let target = Double(ratio.0) / Double(ratio.1)
            let diff = abs(aspectRatio - target)
            if diff < bestDiff
                || (diff == bestDiff
                    && Double(area) > 0.5 * Double(imageSize * imageSize * ratio.0 * ratio.1))
            {
                bestDiff = diff
                best = ratio
            }
        }
        return best
    }

    private func padToSquare(_ image: CIImage, size: Int) -> CIImage {
        let target = CGSize(width: size, height: size)
        let scale = min(target.width / image.extent.width, target.height / image.extent.height)
        let resized = CGSize(
            width: (image.extent.width * scale).rounded(),
            height: (image.extent.height * scale).rounded())
        let scaled = image.toSRGB().resampled(to: resized, method: .bicubic)
        let pad = config.imageMean.map { value -> CGFloat in
            let linear = CGFloat(Int(value * 255)) / 255
            if linear <= 0.0031308 { return 12.92 * linear }
            return 1.055 * pow(linear, 1 / 2.4) - 0.055
        }
        let color = CIColor(
            red: pad[0], green: pad[1],
            blue: pad[2], alpha: 1)
        let background = CIImage(color: color).cropped(
            to: CGRect(x: 0, y: 0, width: target.width, height: target.height))
        let dx = ((target.width - resized.width) / 2).rounded(.down)
        let dy = ((target.height - resized.height) / 2).rounded(.down)
        return scaled.transformed(by: CGAffineTransform(translationX: dx, y: dy))
            .composited(over: background)
            .cropped(to: CGRect(origin: .zero, size: target))
    }

    private func imageArray(_ image: CIImage, size: Int) -> MLXArray {
        planarImageArray(
            image.toSRGB()
            .resampled(to: CGSize(width: size, height: size), method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
        )
            .asType(.bfloat16)
    }

    private func paddedImageArray(_ image: CIImage, size: Int) -> MLXArray {
        planarImageArray(
            padToSquare(image, size: size)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
        )
            .asType(.bfloat16)
    }

    private func planarImageArray(_ image: CIImage) -> MLXArray {
        let width = Int(image.extent.width.rounded())
        let height = Int(image.extent.height.rounded())
        let components = 4
        var rgba = [Float](repeating: 0, count: width * height * components)
        let context = CIContext(options: [.cacheIntermediates: false])
        rgba.withUnsafeMutableBytes { ptr in
            context.render(
                image,
                toBitmap: ptr.baseAddress!,
                rowBytes: width * components * MemoryLayout<Float>.stride,
                bounds: image.extent,
                format: .RGBAf,
                colorSpace: nil)
        }
        var chw = [Float](repeating: 0, count: 3 * width * height)
        let plane = width * height
        let lower = config.imageMean.enumerated().map { -Float($0.element) / Float(config.imageStd[$0.offset]) }
        let upper = config.imageMean.enumerated().map { (1 - Float($0.element)) / Float(config.imageStd[$0.offset]) }
        for i in 0 ..< plane {
            let source = i * components
            chw[i] = min(max(rgba[source], lower[0]), upper[0])
            chw[plane + i] = min(max(rgba[source + 1], lower[1]), upper[1])
            chw[2 * plane + i] = min(max(rgba[source + 2], lower[2]), upper[2])
        }
        return MLXArray(chw, [1, 3, height, width])
    }

    private func dynamicCrops(_ image: CIImage) -> (crops: [CIImage], width: Int, height: Int) {
        let originalWidth = Int(image.extent.width.rounded())
        let originalHeight = Int(image.extent.height.rounded())
        let ratio = closestAspectRatio(width: originalWidth, height: originalHeight, imageSize: cropSize)
        let targetWidth = cropSize * ratio.0
        let targetHeight = cropSize * ratio.1
        let resized = image.toSRGB().resampled(
            to: CGSize(width: targetWidth, height: targetHeight), method: .bicubic)

        var crops: [CIImage] = []
        for i in 0 ..< (ratio.0 * ratio.1) {
            let x = (i % ratio.0) * cropSize
            let row = i / ratio.0
            let y = targetHeight - (row + 1) * cropSize
            let rect = CGRect(x: x, y: y, width: cropSize, height: cropSize)
            crops.append(
                resized.cropped(to: rect)
                    .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY)))
        }
        return (crops, ratio.0, ratio.1)
    }

    private func cubicWeight(_ x: Float) -> Float {
        let a: Float = -0.5
        let x = abs(x)
        if x < 1 {
            return (a + 2) * x * x * x - (a + 3) * x * x + 1
        }
        if x < 2 {
            return a * x * x * x - 5 * a * x * x + 8 * a * x - 4 * a
        }
        return 0
    }

    private func clamp(_ value: Int, lower: Int, upper: Int) -> Int {
        min(max(value, lower), upper)
    }

    private func resizeWeights(sourceSize: Int, targetSize: Int) -> [([(Int, Float)], Float)] {
        let scale = Float(targetSize) / Float(sourceSize)
        let filterScale = scale < 1 ? 1 / scale : 1
        let support = 2 * filterScale
        let inverseScale = Float(sourceSize) / Float(targetSize)
        return (0 ..< targetSize).map { target in
            let center = (Float(target) + 0.5) * inverseScale
            // Match PIL's antialiased bicubic: only include source pixels in the
            // valid range [0, sourceSize). Clamping the start/end (instead of
            // clamping each source index) avoids double-counting boundary
            // pixels and matches the per-row kernel sums that PIL's
            // precompute_coeffs produces.
            let start = max(0, Int(floor(center - support + 0.5)))
            let end = min(sourceSize, Int(floor(center + support + 0.5)))
            var weights: [(Int, Float)] = []
            var sum: Float = 0
            for source in start ..< end {
                let weight = cubicWeight(((Float(source) + 0.5) - center) / filterScale)
                guard weight != 0 else { continue }
                weights.append((source, weight))
                sum += weight
            }
            return (weights, sum == 0 ? 1 : sum)
        }
    }

    private func sourceRGBABytes(_ image: CIImage) -> (bytes: [UInt8], width: Int, height: Int)? {
        let width = Int(image.extent.width.rounded())
        let height = Int(image.extent.height.rounded())
        let context = CIContext(options: [.cacheIntermediates: false])
        guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else { return nil }
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        bytes.withUnsafeMutableBytes { ptr in
            context.render(
                image,
                toBitmap: ptr.baseAddress!,
                rowBytes: width * 4,
                bounds: image.extent,
                format: .RGBA8,
                colorSpace: colorSpace)
        }
        return (bytes, width, height)
    }

    private func cpuBicubicCropPixels(_ image: CIImage, widthCrops: Int, heightCrops: Int) -> MLXArray? {
        guard let source = sourceRGBABytes(image) else { return nil }
        let targetWidth = cropSize * widthCrops
        let targetHeight = cropSize * heightCrops
        let cropCount = widthCrops * heightCrops
        let plane = cropSize * cropSize
        var output = [Float](repeating: 0, count: cropCount * 3 * plane)
        let xWeights = resizeWeights(sourceSize: source.width, targetSize: targetWidth)
        let yWeights = resizeWeights(sourceSize: source.height, targetSize: targetHeight)

        for cropIndex in 0 ..< cropCount {
            let cropX = cropIndex % widthCrops
            let cropY = cropIndex / widthCrops
            let cropBase = cropIndex * 3 * plane
            for y in 0 ..< cropSize {
                let targetY = cropY * cropSize + y
                let (ys, ySum) = yWeights[targetY]
                for x in 0 ..< cropSize {
                    let targetX = cropX * cropSize + x
                    let (xs, xSum) = xWeights[targetX]

                    var rgb = [Float](repeating: 0, count: 3)
                    for (yIndex, yWeight) in ys {
                        for (xIndex, xWeight) in xs {
                            let weight = (yWeight / ySum) * (xWeight / xSum)
                            let sourceOffset = (yIndex * source.width + xIndex) * 4
                            rgb[0] += Float(source.bytes[sourceOffset]) * weight
                            rgb[1] += Float(source.bytes[sourceOffset + 1]) * weight
                            rgb[2] += Float(source.bytes[sourceOffset + 2]) * weight
                        }
                    }
                    let pixel = y * cropSize + x
                    for channel in 0 ..< 3 {
                        let value = round(min(max(rgb[channel], 0), 255)) / 255
                        output[cropBase + channel * plane + pixel] =
                            (value - Float(config.imageMean[channel])) / Float(config.imageStd[channel])
                    }
                }
            }
        }
        return MLXArray(output, [cropCount, 3, cropSize, cropSize]).asType(.bfloat16)
    }

    private func cpuPaddedPixels(_ image: CIImage, size: Int) -> MLXArray? {
        guard let source = sourceRGBABytes(image) else { return nil }
        let scale = min(Float(size) / Float(source.width), Float(size) / Float(source.height))
        let resizedWidth = Int((Float(source.width) * scale).rounded())
        let resizedHeight = Int((Float(source.height) * scale).rounded())
        let offsetX = (size - resizedWidth) / 2
        let offsetY = (size - resizedHeight) / 2
        let xWeights = resizeWeights(sourceSize: source.width, targetSize: resizedWidth)
        let yWeights = resizeWeights(sourceSize: source.height, targetSize: resizedHeight)
        let plane = size * size
        let pad = config.imageMean.map { Float(Int(Float($0) * 255)) / 255 }
        var output = [Float](repeating: 0, count: 3 * plane)

        for y in 0 ..< size {
            for x in 0 ..< size {
                let pixel = y * size + x
                let localX = x - offsetX
                let localY = y - offsetY
                var rgb = pad
                if localX >= 0 && localX < resizedWidth && localY >= 0 && localY < resizedHeight {
                    let (xs, xSum) = xWeights[localX]
                    let (ys, ySum) = yWeights[localY]
                    rgb = [0, 0, 0]
                    for (yIndex, yWeight) in ys {
                        for (xIndex, xWeight) in xs {
                            let weight = (yWeight / ySum) * (xWeight / xSum)
                            let sourceOffset = (yIndex * source.width + xIndex) * 4
                            rgb[0] += Float(source.bytes[sourceOffset]) * weight
                            rgb[1] += Float(source.bytes[sourceOffset + 1]) * weight
                            rgb[2] += Float(source.bytes[sourceOffset + 2]) * weight
                        }
                    }
                    for channel in 0 ..< 3 {
                        rgb[channel] = round(min(max(rgb[channel], 0), 255)) / 255
                    }
                }
                for channel in 0 ..< 3 {
                    output[channel * plane + pixel] =
                        (rgb[channel] - Float(config.imageMean[channel])) / Float(config.imageStd[channel])
                }
            }
        }

        return MLXArray(output, [1, 3, size, size]).asType(.bfloat16)
    }

    private func tokenizeWithImage(prompt: String, image: CIImage) throws -> (
        tokens: [Int], mask: [Bool], cropPixels: MLXArray, globalPixels: MLXArray, spatialCrops: MLXArray
    ) {
        guard prompt.components(separatedBy: config.imageToken).count - 1 == 1 else {
            throw VLMError.processing("DeepSeekOCR prompt must contain exactly one <image> token.")
        }

        let pieces = prompt.components(separatedBy: config.imageToken)
        let imageToken = imageTokenId
        var tokens = tokenizer.encode(text: pieces[0], addSpecialTokens: false)
        var sequenceMask = Array(repeating: false, count: tokens.count)

        let width = Int(image.extent.width.rounded())
        let height = Int(image.extent.height.rounded())
        let cropInfo: (crops: [CIImage], width: Int, height: Int)
        if width <= cropSize && height <= cropSize {
            cropInfo = ([], 1, 1)
        } else {
            cropInfo = dynamicCrops(image)
        }

        let globalPixels = cpuPaddedPixels(image, size: baseSize) ?? paddedImageArray(image, size: baseSize)
        let cropPixels: MLXArray
        if cropInfo.crops.isEmpty {
            cropPixels = MLXArray.zeros([1, 3, baseSize, baseSize]).asType(.bfloat16)
        } else {
            cropPixels =
                cpuBicubicCropPixels(
                    image, widthCrops: cropInfo.width, heightCrops: cropInfo.height)
                ?? concatenated(cropInfo.crops.map { imageArray($0, size: cropSize) }, axis: 0)
        }

        let queries = Int(ceil(Double((cropSize / config.patchSize)) / Double(config.downsampleRatio)))
        let baseQueries = Int(ceil(Double((baseSize / config.patchSize)) / Double(config.downsampleRatio)))
        var imageTokens = Array(repeating: imageToken, count: baseQueries)
        imageTokens.append(imageToken)
        imageTokens = Array(repeating: imageTokens, count: baseQueries).flatMap { $0 }
        imageTokens.append(imageToken)
        if cropInfo.width > 1 || cropInfo.height > 1 {
            var row = Array(repeating: imageToken, count: queries * cropInfo.width)
            row.append(imageToken)
            imageTokens.append(contentsOf: Array(repeating: row, count: queries * cropInfo.height).flatMap { $0 })
        }

        tokens.append(contentsOf: imageTokens)
        sequenceMask.append(contentsOf: Array(repeating: true, count: imageTokens.count))

        let tail = tokenizer.encode(text: pieces[1], addSpecialTokens: false)
        tokens.append(contentsOf: tail)
        sequenceMask.append(contentsOf: Array(repeating: false, count: tail.count))

        tokens.insert(bosTokenId, at: 0)
        sequenceMask.insert(false, at: 0)

        // Match Python inference_mode=True.
        if !tokens.isEmpty {
            tokens.removeLast()
            sequenceMask.removeLast()
        }

        return (
            tokens,
            sequenceMask,
            cropPixels,
            globalPixels,
            MLXArray([Int32(cropInfo.width), Int32(cropInfo.height)]).reshaped(1, 2)
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty, input.audios.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }
        guard input.images.count == 1 else {
            if input.images.isEmpty { throw VLMError.imageRequired }
            throw VLMError.singleImageAllowed
        }

        let rawPrompt = input.prompt.description

        let prompt = try normalizePrompt(rawPrompt, imageCount: input.images.count)
        let image = MediaProcessing.apply(try input.images[0].asCIImage(), processing: input.processing)
        let prepared = try tokenizeWithImage(prompt: prompt, image: image)
        let tokenArray = MLXArray(prepared.tokens).expandedDimensions(axis: 0)
        let attentionMask = (tokenArray .!= padTokenId).asType(.int8)
        let imageMask = MLXArray(prepared.mask.map { $0 ? Int32(1) : Int32(0) })
            .asType(.bool)
            .expandedDimensions(axis: 0)

        return LMInput(
            text: .init(tokens: tokenArray, mask: attentionMask),
            image: .init(
                pixels: prepared.globalPixels,
                cropPixels: prepared.cropPixels,
                sequenceMask: imageMask,
                spatialCrops: prepared.spatialCrops))
    }
}

// MARK: - Model Configuration

public struct DeepseekOCRConfiguration: Decodable, ModelConfigurationValidating, Sendable {
    public let modelType: String
    public let imageTokenIndex: Int
    public let tileTag: String
    public let globalViewPos: String
    public let textConfig: TextConfiguration
    public let visionConfig: VisionConfiguration
    public let projectorConfig: ProjectorConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case imageTokenIndex = "image_token_index"
        case tileTag = "tile_tag"
        case globalViewPos = "global_view_pos"
        case textConfig = "text_config"
        case languageConfig = "language_config"
        case visionConfig = "vision_config"
        case projectorConfig = "projector_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 128815
        tileTag = try c.decodeIfPresent(String.self, forKey: .tileTag) ?? "2D"
        globalViewPos = try c.decodeIfPresent(String.self, forKey: .globalViewPos) ?? "head"
        if let text = try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfig) {
            textConfig = text
        } else {
            textConfig = try c.decode(TextConfiguration.self, forKey: .languageConfig)
        }
        visionConfig = try c.decode(VisionConfiguration.self, forKey: .visionConfig)
        projectorConfig = try c.decode(ProjectorConfiguration.self, forKey: .projectorConfig)
    }

    public func validateModelConfiguration() throws {
        guard modelType == "deepseekocr" else {
            throw ModelFactoryError.unsupportedModelType(modelType)
        }
        guard tileTag == "2D" else {
            throw ModelFactoryError.invalidConfiguration("DeepSeekOCR only supports tile_tag=2D")
        }
        guard projectorConfig.projectorType == "linear" else {
            throw ModelFactoryError.invalidConfiguration(
                "DeepSeekOCR only supports linear projector for this checkpoint")
        }
        guard textConfig.qkNopeHeadDim == 0, textConfig.qkRopeHeadDim == 0 else {
            throw ModelFactoryError.invalidConfiguration(
                "DeepSeekOCR Swift port supports the LlamaAttention path only")
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let vocabSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let moeIntermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let nSharedExperts: Int?
        public let nRoutedExperts: Int?
        public let routedScalingFactor: Float
        public let numExpertsPerTok: Int
        public let firstKDenseReplace: Int
        public let moeLayerFreq: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let attentionBias: Bool
        public let scoringFunc: String
        public let topkMethod: String
        public let qkNopeHeadDim: Int
        public let qkRopeHeadDim: Int

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case nSharedExperts = "n_shared_experts"
            case nRoutedExperts = "n_routed_experts"
            case routedScalingFactor = "routed_scaling_factor"
            case numExpertsPerTok = "num_experts_per_tok"
            case firstKDenseReplace = "first_k_dense_replace"
            case moeLayerFreq = "moe_layer_freq"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case attentionBias = "attention_bias"
            case scoringFunc = "scoring_func"
            case topkMethod = "topk_method"
            case qkNopeHeadDim = "qk_nope_head_dim"
            case qkRopeHeadDim = "qk_rope_head_dim"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            vocabSize = try c.decode(Int.self, forKey: .vocabSize)
            hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
            intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
            moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
            numHiddenLayers = try c.decode(Int.self, forKey: .numHiddenLayers)
            numAttentionHeads = try c.decode(Int.self, forKey: .numAttentionHeads)
            numKeyValueHeads = try c.decode(Int.self, forKey: .numKeyValueHeads)
            nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
            nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
            routedScalingFactor =
                try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
            numExpertsPerTok = try c.decode(Int.self, forKey: .numExpertsPerTok)
            firstKDenseReplace = try c.decode(Int.self, forKey: .firstKDenseReplace)
            moeLayerFreq = try c.decodeIfPresent(Int.self, forKey: .moeLayerFreq) ?? 1
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
            ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            scoringFunc = try c.decodeIfPresent(String.self, forKey: .scoringFunc) ?? "softmax"
            topkMethod = try c.decodeIfPresent(String.self, forKey: .topkMethod) ?? "greedy"
            qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
            qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let layers: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        public let numChannels: Int
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case layers
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case numChannels = "num_channels"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "vision"
            layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 24
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
            numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 224
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }
    }

    public struct ProjectorConfiguration: Decodable, Sendable {
        public let projectorType: String
        public let inputDim: Int
        public let nEmbed: Int

        enum CodingKeys: String, CodingKey {
            case projectorType = "projector_type"
            case inputDim = "input_dim"
            case nEmbed = "n_embed"
        }
    }
}

// MARK: - Text Model

private final class DeepseekOCRAttention: Module {
    let config: DeepseekOCRConfiguration.TextConfiguration
    let headDim: Int
    let scale: Float
    let rope: RoPELayer

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        self.config = config
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)
        self.rope = RoPE(
            dimensions: headDim,
            traditional: config.ropeTraditional,
            base: config.ropeTheta)
        self._qProj.wrappedValue = Linear(
            config.hiddenSize, config.numAttentionHeads * headDim, bias: config.attentionBias)
        self._kProj.wrappedValue = Linear(
            config.hiddenSize, config.numKeyValueHeads * headDim, bias: config.attentionBias)
        self._vProj.wrappedValue = Linear(
            config.hiddenSize, config.numKeyValueHeads * headDim, bias: config.attentionBias)
        self._oProj.wrappedValue = Linear(
            config.numAttentionHeads * headDim, config.hiddenSize, bias: config.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        var queries = qProj(x).reshaped(batch, length, config.numAttentionHeads, headDim)
            .transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(batch, length, config.numKeyValueHeads, headDim)
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, config.numKeyValueHeads, headDim)
            .transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
        keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)
        return oProj(output)
    }
}

private final class DeepseekOCRMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class DeepseekOCRMoEGate: Module {
    let config: DeepseekOCRConfiguration.TextConfiguration
    @ModuleInfo(key: "weight") var weight: MLXArray

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        self.config = config
        self._weight.wrappedValue = MLXArray.zeros([
            config.nRoutedExperts ?? 0, config.hiddenSize,
        ])
    }

    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var scores = matmul(x, weight.transposed())
        if config.scoringFunc == "softmax" {
            scores = softmax(scores, axis: -1, precise: true)
        } else {
            scores = sigmoid(scores)
        }
        let k = config.numExpertsPerTok
        let indices = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
        let selected = takeAlong(scores, indices, axis: -1) * config.routedScalingFactor
        return (indices, selected)
    }
}

private final class DeepseekOCRMoE: Module, UnaryLayer {
    let config: DeepseekOCRConfiguration.TextConfiguration
    @ModuleInfo(key: "gate") var gate: DeepseekOCRMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekOCRMLP?

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        self.config = config
        self._gate.wrappedValue = DeepseekOCRMoEGate(config)
        self._switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts ?? 0,
            bias: false)
        if let shared = config.nSharedExperts {
            self._sharedExperts.wrappedValue = DeepseekOCRMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.moeIntermediateSize * shared)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = weightedExpertSum(y, scores)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

private final class DeepseekOCRDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DeepseekOCRAttention
    @ModuleInfo(key: "mlp") var mlp: Module
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: DeepseekOCRConfiguration.TextConfiguration, layerIndex: Int) {
        self._attention.wrappedValue = DeepseekOCRAttention(config)
        if config.nRoutedExperts != nil
            && layerIndex >= config.firstKDenseReplace
            && layerIndex % config.moeLayerFreq == 0
        {
            self._mlp.wrappedValue = DeepseekOCRMoE(config)
        } else {
            self._mlp.wrappedValue = DeepseekOCRMLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize)
        }
        self._inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        let mlpLayer = mlp as! any UnaryLayer
        return h + mlpLayer(postAttentionLayerNorm(h))
    }
}

private final class DeepseekOCRTextInnerModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [DeepseekOCRDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map {
            DeepseekOCRDecoderLayer(config, layerIndex: $0)
        }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ inputs: MLXArray?, inputsEmbeds: MLXArray? = nil, cache: [KVCache]? = nil
    ) -> MLXArray {
        var h = inputsEmbeds ?? embedTokens(inputs!)
        let mask = createAttentionMask(h: h, cache: cache?.first)
        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }
        return norm(h)
    }
}

private final class DeepseekOCRLanguageModel: Module {
    @ModuleInfo var model: DeepseekOCRTextInnerModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: DeepseekOCRConfiguration.TextConfiguration) {
        self.model = DeepseekOCRTextInnerModel(config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    func callAsFunction(
        _ inputs: MLXArray?, cache: [KVCache]? = nil, inputsEmbeds: MLXArray? = nil
    ) -> LMOutput {
        LMOutput(logits: lmHead(model(inputs, inputsEmbeds: inputsEmbeds, cache: cache)))
    }
}

private final class DeepseekOCRProjector: Module, UnaryLayer {
    @ModuleInfo var layers: Linear

    init(_ config: DeepseekOCRConfiguration.ProjectorConfiguration) {
        self.layers = Linear(config.inputDim, config.nEmbed, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers(x)
    }
}

// MARK: - Vision Model

private final class DeepseekOCRVisionAttention: Module {
    let heads: Int
    let scale: Float
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        self.scale = pow(Float(dimensions / heads), -0.5)
        self._qkvProj.wrappedValue = Linear(dimensions, dimensions * 3, bias: true)
        self._outProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let qkv = qkvProj(x)
        var parts = split(qkv, parts: 3, axis: -1)
        let (batch, length) = (x.dim(0), x.dim(1))
        for i in 0 ..< 3 {
            parts[i] = parts[i].reshaped(batch, length, heads, -1).transposed(0, 2, 1, 3)
        }
        let output = MLXFast.scaledDotProductAttention(
            queries: parts[0], keys: parts[1], values: parts[2], scale: scale, mask: nil
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, length, -1)
        return outProj(output)
    }
}

private final class DeepseekOCRVisionMLP: Module, UnaryLayer {
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    let gelu = GELU()

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self.fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: true)
        self.fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fc2(gelu(fc1(x)))
    }
}

private final class DeepseekOCRVisionLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttention: DeepseekOCRVisionAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo var mlp: DeepseekOCRVisionMLP
    @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self._selfAttention.wrappedValue = DeepseekOCRVisionAttention(
            dimensions: config.hiddenSize, heads: config.numAttentionHeads)
        self._layerNorm1.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.mlp = DeepseekOCRVisionMLP(config)
        self._layerNorm2.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + selfAttention(layerNorm1(x))
        return h + mlp(layerNorm2(h))
    }
}

private final class DeepseekOCRVisionEmbeddings: Module {
    @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
    @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

    let config: DeepseekOCRConfiguration.VisionConfiguration
    let baseImageSize = 224
    let numPositions: Int

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self.config = config
        self._classEmbedding.wrappedValue = MLXRandom.normal([config.hiddenSize])
        self._patchEmbedding.wrappedValue = Conv2d(
            inputChannels: config.numChannels,
            outputChannels: config.hiddenSize,
            kernelSize: IntOrPair(config.patchSize),
            stride: IntOrPair(config.patchSize),
            bias: false)
        self.numPositions = (baseImageSize / config.patchSize) * (baseImageSize / config.patchSize) + 1
        self._positionEmbedding.wrappedValue = Embedding(
            embeddingCount: numPositions,
            dimensions: config.hiddenSize)
    }

    private func absolutePosition(_ length: Int) -> MLXArray {
        let ids = MLXArray(Array(0 ..< numPositions))[.newAxis, 0...]
        let pos = positionEmbedding(ids)
        guard length != numPositions else { return pos }
        let cls = pos[0..., ..<1, 0...]
        let patch = pos[0..., 1..., 0...]
        let src = Int(Double(numPositions - 1).squareRoot())
        let dst = Int(Double(length - 1).squareRoot())
        let resized = bicubicInterpolate(
            patch.reshaped(1, src, src, config.hiddenSize).transposed(0, 3, 1, 2),
            size: (dst, dst)
        )
        .transposed(0, 2, 3, 1)
        .reshaped(1, dst * dst, config.hiddenSize)
        return concatenated([cls, resized], axis: 1)
    }

    func callAsFunction(_ x: MLXArray, patchEmbeds: MLXArray?) -> MLXArray {
        let batch = x.dim(0)
        let patches = flattened(patchEmbeds ?? patchEmbedding(x), start: 1, end: 2)
        let cls = broadcast(classEmbedding, to: [batch, 1, config.hiddenSize])
        let embeddings = concatenated([cls, patches], axis: 1)
        return embeddings + absolutePosition(embeddings.dim(1)).asType(embeddings.dtype)
    }
}

private final class DeepseekOCRVisionTransformer: Module, UnaryLayer {
    let layers: [DeepseekOCRVisionLayer]

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self.layers = (0 ..< config.layers).map { _ in DeepseekOCRVisionLayer(config) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

private final class DeepseekOCRVisionModel: Module {
    @ModuleInfo var embeddings: DeepseekOCRVisionEmbeddings
    @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
    @ModuleInfo var transformer: DeepseekOCRVisionTransformer

    init(_ config: DeepseekOCRConfiguration.VisionConfiguration) {
        self.embeddings = DeepseekOCRVisionEmbeddings(config)
        self._preLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        self.transformer = DeepseekOCRVisionTransformer(config)
    }

    func callAsFunction(_ x: MLXArray, patchEmbeds: MLXArray? = nil) -> MLXArray {
        transformer(preLayerNorm(embeddings(x, patchEmbeds: patchEmbeds)))
    }
}

// MARK: - SAM Encoder

private final class DeepseekOCRSAMBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var attn: DeepseekOCRSAMAttention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var mlp: DeepseekOCRSAMMLP
    let windowSize: Int

    init(dim: Int, heads: Int, windowSize: Int, inputSize: Int) {
        self.windowSize = windowSize
        self.norm1 = LayerNorm(dimensions: dim, eps: 1e-6)
        self.attn = DeepseekOCRSAMAttention(dim: dim, heads: heads, inputSize: windowSize == 0 ? inputSize : windowSize)
        self.norm2 = LayerNorm(dimensions: dim, eps: 1e-6)
        self.mlp = DeepseekOCRSAMMLP(dim: dim, hidden: dim * 4)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm1(x)
        let height = h.dim(1)
        let width = h.dim(2)
        var paddedHW: (Int, Int)?
        if windowSize > 0 {
            (h, paddedHW) = DeepseekOCRSAMEncoder.windowPartition(h, windowSize: windowSize)
        }
        h = attn(h)
        if let paddedHW {
            h = DeepseekOCRSAMEncoder.windowUnpartition(
                h, windowSize: windowSize, paddedHW: paddedHW, originalHW: (height, width))
        }
        let y = x + h
        return y + mlp(norm2(y))
    }
}

private final class DeepseekOCRSAMMLP: Module, UnaryLayer {
    @ModuleInfo var lin1: Linear
    @ModuleInfo var lin2: Linear
    let gelu = GELU()

    init(dim: Int, hidden: Int) {
        self.lin1 = Linear(dim, hidden)
        self.lin2 = Linear(hidden, dim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lin2(gelu(lin1(x)))
    }
}

private final class DeepseekOCRSAMAttention: Module {
    let heads: Int
    let scale: Float
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    @ModuleInfo(key: "rel_pos_h") var relPosH: MLXArray
    @ModuleInfo(key: "rel_pos_w") var relPosW: MLXArray

    init(dim: Int, heads: Int, inputSize: Int) {
        self.heads = heads
        self.scale = pow(Float(dim / heads), -0.5)
        self.qkv = Linear(dim, dim * 3, bias: true)
        self.proj = Linear(dim, dim)
        let headDim = dim / heads
        self._relPosH.wrappedValue = MLXArray.zeros([2 * inputSize - 1, headDim])
        self._relPosW.wrappedValue = MLXArray.zeros([2 * inputSize - 1, headDim])
    }

    private func relativePosition(_ querySize: Int, _ keySize: Int, _ relPos: MLXArray) -> MLXArray {
        let maxRelDist = 2 * max(querySize, keySize) - 1
        var resized = relPos
        if relPos.dim(0) != maxRelDist {
            let sourceLength = relPos.dim(0)
            let indices =
                MLXArray.arange(maxRelDist, dtype: .float32) * (Float(sourceLength) / Float(maxRelDist))
            let lower = floor(indices).asType(.int32)
            let upper = minimum(lower + 1, MLXArray(Int32(sourceLength - 1)))
            let weight = indices - lower.asType(.float32)
            let lowerValues = take(relPos.asType(.float32), lower, axis: 0)
            let upperValues = take(relPos.asType(.float32), upper, axis: 0)
            resized =
                (lowerValues * (1 - weight[0..., .newAxis])
                    + upperValues * weight[0..., .newAxis])
                .asType(relPos.dtype)
        }

        let queryCoords =
            MLXArray.arange(querySize, dtype: .float32)[0..., .newAxis]
            * max(Float(keySize) / Float(querySize), 1)
        let keyCoords =
            MLXArray.arange(keySize, dtype: .float32)[.newAxis, 0...]
            * max(Float(querySize) / Float(keySize), 1)
        let relativeCoords =
            (queryCoords - keyCoords)
            + Float(keySize - 1) * max(Float(querySize) / Float(keySize), 1)
        return take(resized, relativeCoords.asType(.int32), axis: 0)
    }

    private func decomposedRelativePositionBias(
        queries: MLXArray, height: Int, width: Int
    ) -> MLXArray {
        let query = queries.reshaped(queries.dim(0) * queries.dim(1), height * width, -1)
        let relH = relativePosition(height, height, relPosH)
        let relW = relativePosition(width, width, relPosW)
        let reshapedQuery = query.reshaped(query.dim(0), height, width, -1)
        var heightBias = einsum("bhwc,hkc->bhwk", reshapedQuery, relH)
        var widthBias = einsum("bhwc,wkc->bhwk", reshapedQuery, relW)
        heightBias = heightBias[0..., 0..., 0..., 0..., .newAxis]
            .reshaped(query.dim(0), height * width, height, 1)
        widthBias = widthBias[0..., 0..., 0..., .newAxis, 0...]
            .reshaped(query.dim(0), height * width, 1, width)
        return (heightBias + widthBias)
            .reshaped(queries.dim(0), queries.dim(1), height * width, height * width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (batch, height, width) = (x.dim(0), x.dim(1), x.dim(2))
        let qkv = qkv(x).reshaped(batch, height * width, 3, heads, -1)
            .transposed(2, 0, 3, 1, 4)
        let queries = qkv[0].reshaped(batch, heads, height * width, -1)
        let keys = qkv[1].reshaped(batch, heads, height * width, -1)
        let values = qkv[2].reshaped(batch, heads, height * width, -1)
        let attentionBias = decomposedRelativePositionBias(
            queries: queries, height: height, width: width)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: attentionBias
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batch, height, width, -1)
        return proj(output)
    }
}

private final class DeepseekOCRSAMEncoder: Module {
    @ModuleInfo(key: "patch_embed") var patchEmbed: DeepseekOCRSAMPatchEmbed
    @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray
    let blocks: [DeepseekOCRSAMBlock]
    @ModuleInfo var neck: [Module]
    @ModuleInfo(key: "net_2") var net2: Conv2d
    @ModuleInfo(key: "net_3") var net3: Conv2d

    override init() {
        let imageSize = 1024
        let patchSize = 16
        let width = 768
        self._patchEmbed.wrappedValue = DeepseekOCRSAMPatchEmbed(
            inChannels: 3, embedDim: width, patchSize: patchSize)
        self._posEmbed.wrappedValue = MLXArray.zeros([1, imageSize / patchSize, imageSize / patchSize, width])
        let global = Set([2, 5, 8, 11])
        self.blocks = (0 ..< 12).map {
            DeepseekOCRSAMBlock(
                dim: width, heads: 12, windowSize: global.contains($0) ? 0 : 14,
                inputSize: imageSize / patchSize)
        }
        self.neck = [
            Conv2d(inputChannels: width, outputChannels: 256, kernelSize: 1, bias: false),
            LayerNorm(dimensions: 256, eps: 1e-6),
            Conv2d(inputChannels: 256, outputChannels: 256, kernelSize: 3, padding: 1, bias: false),
            LayerNorm(dimensions: 256, eps: 1e-6),
        ]
        self._net2.wrappedValue = Conv2d(
            inputChannels: 256, outputChannels: 512, kernelSize: 3, stride: 2, padding: 1,
            bias: false)
        self._net3.wrappedValue = Conv2d(
            inputChannels: 512, outputChannels: 1024, kernelSize: 3, stride: 2, padding: 1,
            bias: false)
    }

    static func windowPartition(_ x: MLXArray, windowSize: Int) -> (MLXArray, (Int, Int)) {
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let padH = (windowSize - height % windowSize) % windowSize
        let padW = (windowSize - width % windowSize) % windowSize
        var y = x
        if padH > 0 || padW > 0 {
            y = padded(y, widths: [0, [0, padH], [0, padW], 0])
        }
        let hp = height + padH
        let wp = width + padW
        y = y.reshaped(batch, hp / windowSize, windowSize, wp / windowSize, windowSize, channels)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(-1, windowSize, windowSize, channels)
        return (y, (hp, wp))
    }

    static func windowUnpartition(
        _ windows: MLXArray, windowSize: Int, paddedHW: (Int, Int), originalHW: (Int, Int)
    ) -> MLXArray {
        let (hp, wp) = paddedHW
        let (height, width) = originalHW
        let batch = windows.dim(0) / ((hp / windowSize) * (wp / windowSize))
        var x = windows.reshaped(batch, hp / windowSize, wp / windowSize, windowSize, windowSize, -1)
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batch, hp, wp, -1)
        if hp > height || wp > width {
            x = x[0..., ..<height, ..<width, 0...]
        }
        return x
    }

    private func absolutePosition(_ target: Int) -> MLXArray {
        guard posEmbed.dim(1) != target else { return posEmbed }
        return bicubicInterpolate(posEmbed.transposed(0, 3, 1, 2), size: (target, target))
            .transposed(0, 2, 3, 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = patchEmbed(x) + absolutePosition(patchEmbed(x).dim(1))
        for block in blocks {
            h = block(h)
        }
        for layer in neck {
            h = (layer as! any UnaryLayer)(h)
        }
        h = net2(h)
        h = net3(h)
        return h
    }
}

private final class DeepseekOCRSAMPatchEmbed: Module, UnaryLayer {
    @ModuleInfo var proj: Conv2d

    init(inChannels: Int, embedDim: Int, patchSize: Int) {
        self.proj = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: IntOrPair(patchSize),
            stride: IntOrPair(patchSize))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj(x)
    }
}

// MARK: - DeepSeekOCR

public final class DeepseekOCR: Module, VLMModel, KVCacheDimensionProvider {
    public let config: DeepseekOCRConfiguration

    @ModuleInfo(key: "vision_model") private var visionModel: DeepseekOCRVisionModel
    @ModuleInfo(key: "sam_model") private var samModel: DeepseekOCRSAMEncoder
    @ModuleInfo(key: "language_model") private var languageModel: DeepseekOCRLanguageModel
    @ModuleInfo private var projector: DeepseekOCRProjector
    @ModuleInfo(key: "image_newline") var imageNewline: MLXArray
    @ModuleInfo(key: "view_separator") var viewSeparator: MLXArray

    public init(_ config: DeepseekOCRConfiguration) {
        self.config = config
        self._visionModel.wrappedValue = DeepseekOCRVisionModel(config.visionConfig)
        self._samModel.wrappedValue = DeepseekOCRSAMEncoder()
        self._languageModel.wrappedValue = DeepseekOCRLanguageModel(config.textConfig)
        self.projector = DeepseekOCRProjector(config.projectorConfig)
        let embedStd = 1 / sqrt(Float(config.projectorConfig.nEmbed))
        self._imageNewline.wrappedValue =
            MLXRandom.normal([config.projectorConfig.nEmbed]) * embedStd
        self._viewSeparator.wrappedValue =
            MLXRandom.normal([config.projectorConfig.nEmbed]) * embedStd
        super.init()
    }

    public var kvHeads: [Int] {
        Array(repeating: config.textConfig.numKeyValueHeads, count: config.textConfig.numHiddenLayers)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func imageTokenIndices(_ mask: MLXArray, row: Int) -> [Int] {
        mask[row].asArray(Bool.self).enumerated().compactMap { $0.element ? $0.offset : nil }
    }

    private func appendNewlines(_ features: MLXArray, rows: Int, cols: Int) -> MLXArray {
        let dim = features.dim(-1)
        let grid = features.reshaped(rows, cols, dim)
        let newline = broadcast(imageNewline[.newAxis, .newAxis, 0...], to: [rows, 1, dim])
        return concatenated([grid, newline], axis: 1).reshaped(-1, dim)
    }

    private func visualFeatures(globalPixels: MLXArray, cropPixels: MLXArray?, spatialCrops: MLXArray)
        -> [MLXArray]
    {
        let cropShapes = spatialCrops.asArray(Int32.self)
        var outputs: [MLXArray] = []
        var patchOffset = 0
        for row in 0 ..< spatialCrops.dim(0) {
            let widthCrops = Int(cropShapes[row * 2])
            let heightCrops = Int(cropShapes[row * 2 + 1])
            let global = globalPixels[row ..< (row + 1)].transposed(0, 2, 3, 1)
            let globalSAM = samModel(global)
            let globalVision = visionModel(global, patchEmbeds: globalSAM)
            var globalFeatures = concatenated(
                [globalVision[0..., 1..., 0...], flattened(globalSAM, start: 1, end: 2)],
                axis: -1)
            globalFeatures = projector(globalFeatures)[0]
            let gSide = Int(Double(globalFeatures.dim(0)).squareRoot())
            globalFeatures = appendNewlines(globalFeatures, rows: gSide, cols: gSide)

            if widthCrops > 1 || heightCrops > 1, let cropPixels {
                let count = widthCrops * heightCrops
                let crops = cropPixels[patchOffset ..< (patchOffset + count)].transposed(0, 2, 3, 1)
                patchOffset += count
                let localSAM = samModel(crops)
                let localVision = visionModel(crops, patchEmbeds: localSAM)
                var localFeatures = concatenated(
                    [localVision[0..., 1..., 0...], flattened(localSAM, start: 1, end: 2)],
                    axis: -1)
                localFeatures = projector(localFeatures)
                let localSide = Int(Double(localFeatures.dim(1)).squareRoot())
                let dim = localFeatures.dim(-1)
                localFeatures = localFeatures
                    .reshaped(heightCrops, widthCrops, localSide, localSide, dim)
                    .transposed(0, 2, 1, 3, 4)
                    .reshaped(heightCrops * localSide, widthCrops * localSide, dim)
                let newline = broadcast(
                    imageNewline[.newAxis, .newAxis, 0...],
                    to: [heightCrops * localSide, 1, dim])
                localFeatures = concatenated([localFeatures, newline], axis: 1).reshaped(-1, dim)
                outputs.append(concatenated([localFeatures, globalFeatures, viewSeparator[.newAxis, 0...]], axis: 0))
            } else {
                outputs.append(concatenated([globalFeatures, viewSeparator[.newAxis, 0...]], axis: 0))
            }
        }
        return outputs
    }

    private func inputEmbeddings(input: LMInput) throws -> MLXArray {
        let inputEmbeds = languageModel.model.embedTokens(input.text.tokens)
        guard input.text.tokens.dim(1) != 1,
            let image = input.image,
            let sequenceMask = image.sequenceMask,
            let spatialCrops = image.spatialCrops
        else {
            return inputEmbeds
        }
        let features = visualFeatures(
            globalPixels: image.pixels.asType(inputEmbeds.dtype),
            cropPixels: image.cropPixels?.asType(inputEmbeds.dtype),
            spatialCrops: spatialCrops)
        for (row, rowFeatures) in features.enumerated() {
            let indices = imageTokenIndices(sequenceMask, row: row)
            guard indices.count == rowFeatures.dim(0) else {
                throw VLMError.processing(
                    "DeepSeekOCR image features and image token count mismatch: \(rowFeatures.dim(0)) != \(indices.count)")
            }
            inputEmbeds[row, MLXArray(indices), 0...] = rowFeatures
        }
        return inputEmbeds
    }

    public func prepare(_ input: LMInput, cache: [any KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let embeddings = try inputEmbeddings(input: input)
        let totalPositions = embeddings.dim(1)
        let prefillStepSize = max(windowSize ?? totalPositions, totalPositions)
        var processed = 0
        while totalPositions - processed > 1 {
            let chunkLength = min(prefillStepSize, totalPositions - processed - 1)
            let range = processed ..< (processed + chunkLength)
            _ = languageModel(nil, cache: cache, inputsEmbeds: embeddings[0..., range, 0...])
            asyncEval(cache)
            processed += chunkLength
        }
        eval(cache)
        let result = languageModel(
            nil, cache: cache, inputsEmbeds: embeddings[0..., processed..., 0...])
        return .logits(result)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        languageModel(input.tokens, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var transformed: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("position_ids") { continue }
            var k = key
            if k.contains("model.layers"), !k.contains("language_model") {
                k = k.replacingOccurrences(of: "model.layers", with: "language_model.model.layers")
            }
            if k.contains("model.embed_tokens"), !k.contains("language_model") {
                k = k.replacingOccurrences(
                    of: "model.embed_tokens", with: "language_model.model.embed_tokens")
            }
            if k.contains("model.norm"), !k.contains("language_model") {
                k = k.replacingOccurrences(of: "model.norm", with: "language_model.model.norm")
            }
            if k.contains("model.vision_model") {
                k = k.replacingOccurrences(of: "model.vision_model", with: "vision_model")
            }
            if k.contains("model.sam_model") {
                k = k.replacingOccurrences(of: "model.sam_model", with: "sam_model")
            }
            if k.contains("model.projector") {
                k = k.replacingOccurrences(of: "model.projector", with: "projector")
            }
            if k.contains("model.view_seperator") {
                k = k.replacingOccurrences(of: "model.view_seperator", with: "view_separator")
            }
            if k.contains("model.image_newline") {
                k = k.replacingOccurrences(of: "model.image_newline", with: "image_newline")
            }
            if k == "lm_head.weight" {
                k = "language_model.lm_head.weight"
            }
            transformed[k] = value
        }
        return sanitizeExperts(weights: transformed)
    }

    private func sanitizeExperts(weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        guard let experts = config.textConfig.nRoutedExperts else { return weights }
        for layer in 0 ..< config.textConfig.numHiddenLayers {
            let prefix = "language_model.model.layers.\(layer)"
            for (old, new) in [("gate_proj", "gate_proj"), ("down_proj", "down_proj"), ("up_proj", "up_proj")] {
                for key in ["weight", "scales", "biases"] {
                    let first = "\(prefix).mlp.experts.0.\(old).\(key)"
                    if weights[first] != nil {
                        let joined = (0 ..< experts).map {
                            weights.removeValue(forKey: "\(prefix).mlp.experts.\($0).\(old).\(key)")!
                        }
                        weights["\(prefix).mlp.switch_mlp.\(new).\(key)"] = stacked(joined)
                    }
                }
            }
        }
        return weights
    }
}
