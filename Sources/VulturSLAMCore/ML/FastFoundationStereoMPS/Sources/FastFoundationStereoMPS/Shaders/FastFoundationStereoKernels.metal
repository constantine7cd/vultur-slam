#include <metal_stdlib>
using namespace metal;

struct LayoutConstants {
    uint n;
    uint c;
    uint h;
    uint w;
};

struct CombinedVolumeConstants {
    uint h;
    uint w;
    uint d;
    uint featureChannels;
    uint projectionChannels;
    uint groups;
    uint outputChannels;
};

struct InitialDisparityConstants {
    uint h;
    uint w;
    uint d;
};

struct GeometryLookupConstants {
    uint h;
    uint w;
    uint d;
    uint featureChannels;
    uint volumeChannels;
    uint radius;
    uint levels;
    uint outputChannels;
};

struct ContextUpsampleConstants {
    uint lowH;
    uint lowW;
    uint highH;
    uint highW;
};

struct ElementwiseConstants {
    uint count;
    float scale;
};

kernel void nchw_to_nhwc_fp32(
    device const float* source [[buffer(0)]],
    device float* destination [[buffer(1)]],
    constant LayoutConstants& c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = c.n * c.h * c.w * c.c;
    if (gid >= total) {
        return;
    }

    const uint channel = gid % c.c;
    const uint x = (gid / c.c) % c.w;
    const uint y = (gid / (c.c * c.w)) % c.h;
    const uint batch = gid / (c.c * c.w * c.h);
    const uint sourceIndex = ((batch * c.c + channel) * c.h + y) * c.w + x;
    destination[gid] = source[sourceIndex];
}

static inline float grouped_norm(
    device const float* feature,
    uint y,
    uint x,
    uint width,
    uint featureChannels,
    uint group,
    uint channelsPerGroup
) {
    float sum = 0.0f;
    const uint channelBase = group * channelsPerGroup;
    const uint pixelBase = (y * width + x) * featureChannels;
    for (uint k = 0; k < channelsPerGroup; ++k) {
        const float value = feature[pixelBase + channelBase + k];
        sum += value * value;
    }
    return rsqrt(max(sum, 1.0e-12f));
}

kernel void build_combined_volume_fp32(
    device const float* projectedLeft [[buffer(0)]],
    device const float* projectedRight [[buffer(1)]],
    device const float* leftFeature [[buffer(2)]],
    device const float* rightFeature [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant CombinedVolumeConstants& c [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = c.d * c.h * c.w * c.outputChannels;
    if (gid >= total) {
        return;
    }

    const uint channel = gid % c.outputChannels;
    const uint x = (gid / c.outputChannels) % c.w;
    const uint y = (gid / (c.outputChannels * c.w)) % c.h;
    const uint disp = gid / (c.outputChannels * c.w * c.h);
    const int shiftedX = int(x) - int(disp);

    if (channel < c.groups) {
        if (shiftedX < 0) {
            output[gid] = 0.0f;
            return;
        }
        const uint channelsPerGroup = c.featureChannels / c.groups;
        const uint group = channel;
        const float leftNorm = grouped_norm(leftFeature, y, x, c.w, c.featureChannels, group, channelsPerGroup);
        const float rightNorm = grouped_norm(rightFeature, y, uint(shiftedX), c.w, c.featureChannels, group, channelsPerGroup);
        const uint channelBase = group * channelsPerGroup;
        const uint leftBase = (y * c.w + x) * c.featureChannels;
        const uint rightBase = (y * c.w + uint(shiftedX)) * c.featureChannels;
        float acc = 0.0f;
        for (uint k = 0; k < channelsPerGroup; ++k) {
            acc += leftFeature[leftBase + channelBase + k] * rightFeature[rightBase + channelBase + k];
        }
        output[gid] = acc * leftNorm * rightNorm;
        return;
    }

    const uint concatChannel = channel - c.groups;
    if (concatChannel < c.projectionChannels) {
        output[gid] = projectedLeft[(y * c.w + x) * c.projectionChannels + concatChannel];
    } else {
        const uint projectionChannel = concatChannel - c.projectionChannels;
        output[gid] = shiftedX < 0 ? 0.0f : projectedRight[(y * c.w + uint(shiftedX)) * c.projectionChannels + projectionChannel];
    }
}

kernel void initial_disparity_fp32(
    device const float* logits [[buffer(0)]],
    device float* disparity [[buffer(1)]],
    constant InitialDisparityConstants& c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = c.h * c.w;
    if (gid >= total) {
        return;
    }

    const uint base = gid * c.d;
    float maxLogit = logits[base];
    for (uint disp = 1; disp < c.d; ++disp) {
        maxLogit = max(maxLogit, logits[base + disp]);
    }

    float denominator = 0.0f;
    float weighted = 0.0f;
    for (uint disp = 0; disp < c.d; ++disp) {
        const float probability = exp(logits[base + disp] - maxLogit);
        denominator += probability;
        weighted += probability * float(disp);
    }

    disparity[gid] = weighted / denominator;
}

static inline float sample_linear(float leftValue, bool leftValid, float rightValue, bool rightValid, float x) {
    const float xFloor = floor(x);
    const float xFrac = x - xFloor;
    const float floorContribution = leftValid ? leftValue * (1.0f - xFrac) : 0.0f;
    const float ceilContribution = rightValid ? rightValue * xFrac : 0.0f;
    return floorContribution + ceilContribution;
}

static inline float normalized_dot(
    device const float* leftFeature,
    device const float* rightFeature,
    uint y,
    uint leftX,
    uint rightX,
    constant GeometryLookupConstants& c
) {
    const uint leftBase = (y * c.w + leftX) * c.featureChannels;
    const uint rightBase = (y * c.w + rightX) * c.featureChannels;
    float leftNormSquared = 0.0f;
    float rightNormSquared = 0.0f;
    float dot = 0.0f;
    for (uint channel = 0; channel < c.featureChannels; ++channel) {
        const float leftValue = leftFeature[leftBase + channel];
        const float rightValue = rightFeature[rightBase + channel];
        leftNormSquared += leftValue * leftValue;
        rightNormSquared += rightValue * rightValue;
        dot += leftValue * rightValue;
    }
    return dot * rsqrt(max(leftNormSquared, 1.0e-12f)) * rsqrt(max(rightNormSquared, 1.0e-12f));
}

static inline float corr_level_value(
    device const float* leftFeature,
    device const float* rightFeature,
    uint level,
    uint y,
    uint leftX,
    uint rightX,
    constant GeometryLookupConstants& c
) {
    if (level == 0) {
        return normalized_dot(leftFeature, rightFeature, y, leftX, rightX, c);
    }
    const uint baseRightX = rightX * 2;
    const float first = normalized_dot(leftFeature, rightFeature, y, leftX, baseRightX, c);
    if (baseRightX + 1 >= c.w) {
        return 0.5f * first;
    }
    const float second = normalized_dot(leftFeature, rightFeature, y, leftX, baseRightX + 1, c);
    return 0.5f * (first + second);
}

static inline float sample_corr_level(
    device const float* leftFeature,
    device const float* rightFeature,
    uint level,
    uint y,
    uint leftX,
    float sampleX,
    constant GeometryLookupConstants& c
) {
    const uint levelWidth = c.w >> level;
    const float xFloor = floor(sampleX);
    const float xCeil = xFloor + 1.0f;
    const bool floorValid = xFloor >= 0.0f && xFloor < float(levelWidth);
    const bool ceilValid = xCeil >= 0.0f && xCeil < float(levelWidth);
    const uint floorIndex = uint(clamp(xFloor, 0.0f, float(levelWidth - 1)));
    const uint ceilIndex = uint(clamp(xCeil, 0.0f, float(levelWidth - 1)));
    const float floorValue = floorValid ? corr_level_value(leftFeature, rightFeature, level, y, leftX, floorIndex, c) : 0.0f;
    const float ceilValue = ceilValid ? corr_level_value(leftFeature, rightFeature, level, y, leftX, ceilIndex, c) : 0.0f;
    return sample_linear(floorValue, floorValid, ceilValue, ceilValid, sampleX);
}

static inline float volume_level_value(
    device const float* regularizedVolume,
    uint level,
    uint disparityIndex,
    uint y,
    uint x,
    uint channel,
    constant GeometryLookupConstants& c
) {
    if (level == 0) {
        return regularizedVolume[(((disparityIndex * c.h + y) * c.w + x) * c.volumeChannels) + channel];
    }
    const uint baseDisparity = disparityIndex * 2;
    const float first = regularizedVolume[(((baseDisparity * c.h + y) * c.w + x) * c.volumeChannels) + channel];
    if (baseDisparity + 1 >= c.d) {
        return 0.5f * first;
    }
    const float second = regularizedVolume[((((baseDisparity + 1) * c.h + y) * c.w + x) * c.volumeChannels) + channel];
    return 0.5f * (first + second);
}

static inline float sample_volume_level(
    device const float* regularizedVolume,
    uint level,
    uint y,
    uint x,
    uint channel,
    float sampleDisparity,
    constant GeometryLookupConstants& c
) {
    const uint levelDisparity = c.d >> level;
    const float xFloor = floor(sampleDisparity);
    const float xCeil = xFloor + 1.0f;
    const bool floorValid = xFloor >= 0.0f && xFloor < float(levelDisparity);
    const bool ceilValid = xCeil >= 0.0f && xCeil < float(levelDisparity);
    const uint floorIndex = uint(clamp(xFloor, 0.0f, float(levelDisparity - 1)));
    const uint ceilIndex = uint(clamp(xCeil, 0.0f, float(levelDisparity - 1)));
    const float floorValue = floorValid ? volume_level_value(regularizedVolume, level, floorIndex, y, x, channel, c) : 0.0f;
    const float ceilValue = ceilValid ? volume_level_value(regularizedVolume, level, ceilIndex, y, x, channel, c) : 0.0f;
    return sample_linear(floorValue, floorValid, ceilValue, ceilValid, sampleDisparity);
}

kernel void geometry_lookup_fp32(
    device const float* leftFeature [[buffer(0)]],
    device const float* rightFeature [[buffer(1)]],
    device const float* regularizedVolume [[buffer(2)]],
    device const float* disparity [[buffer(3)]],
    device float* output [[buffer(4)]],
    constant GeometryLookupConstants& c [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = c.h * c.w * c.outputChannels;
    if (gid >= total) {
        return;
    }

    const uint channel = gid % c.outputChannels;
    const uint x = (gid / c.outputChannels) % c.w;
    const uint y = (gid / (c.outputChannels * c.w)) % c.h;
    const uint radiusCount = c.radius * 2 + 1;
    const uint levelChannels = c.volumeChannels * radiusCount + radiusCount;
    const uint level = channel / levelChannels;
    const uint levelOffset = channel - level * levelChannels;
    const uint radiusIndex = levelOffset < c.volumeChannels * radiusCount ? levelOffset % radiusCount : levelOffset - c.volumeChannels * radiusCount;
    const float dx = float(int(radiusIndex) - int(c.radius));
    const float disparityValue = disparity[y * c.w + x];
    const float levelScale = float(1 << level);

    if (levelOffset < c.volumeChannels * radiusCount) {
        const uint volumeChannel = levelOffset / radiusCount;
        output[gid] = sample_volume_level(regularizedVolume, level, y, x, volumeChannel, disparityValue / levelScale + dx, c);
        return;
    }

    output[gid] = sample_corr_level(leftFeature, rightFeature, level, y, x, float(x) / levelScale - disparityValue / levelScale + dx, c);
}

kernel void context_upsample_fp32(
    device const float* disparityLow [[buffer(0)]],
    device const float* upWeights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ContextUpsampleConstants& c [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    const uint total = c.highH * c.highW;
    if (gid >= total) {
        return;
    }

    const uint highX = gid % c.highW;
    const uint highY = gid / c.highW;
    const uint lowX = highX / 4;
    const uint lowY = highY / 4;
    float acc = 0.0f;

    for (uint ky = 0; ky < 3; ++ky) {
        for (uint kx = 0; kx < 3; ++kx) {
            const uint kernelIndex = ky * 3 + kx;
            const int sampleY = int(lowY) + int(ky) - 1;
            const int sampleX = int(lowX) + int(kx) - 1;
            const float weight = upWeights[(highY * c.highW + highX) * 9 + kernelIndex];
            if (sampleY >= 0 && sampleY < int(c.lowH) && sampleX >= 0 && sampleX < int(c.lowW)) {
                acc += disparityLow[uint(sampleY) * c.lowW + uint(sampleX)] * weight;
            }
        }
    }
    output[gid] = acc;
}

kernel void add_fp32(
    device const float* left [[buffer(0)]],
    device const float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant ElementwiseConstants& c [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= c.count) {
        return;
    }
    output[gid] = left[gid] + right[gid];
}

kernel void scale_fp32(
    device const float* source [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ElementwiseConstants& c [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= c.count) {
        return;
    }
    output[gid] = source[gid] * c.scale;
}
