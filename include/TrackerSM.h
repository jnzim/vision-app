#pragma once
#include <chrono>

enum class TrackState
{
    NO_TARGET,   // No valid target
    TRACKING,    // Regular detection
    LOST         // Temporarily missing but still predicting
};

class TrackSM
{
public:
    // Tunable timeouts
    static constexpr int LOST_TIMEOUT_MS = 500;    // TRACKING -> LOST
    static constexpr int DEAD_TIMEOUT_MS = 2000;   // LOST -> NO_TARGET

    TrackSM() : state(TrackState::NO_TARGET) {}

    // Call when a detection is received
    void onDetection(std::chrono::steady_clock::time_point t)
    {
        lastDetectionTime   = t;
        state               = TrackState::TRACKING;
    }

    // Call each frame even without a detection
    void onFrame(std::chrono::steady_clock::time_point now)
    {
        if (state == TrackState::NO_TARGET)
        {
            return;
        }
        auto ageMs = 
        std::chrono::duration_cast<std::chrono::milliseconds>(now - lastDetectionTime).count();

        if (ageMs > DEAD_TIMEOUT_MS)
        {
            state = TrackState::NO_TARGET;
        }
        else if (ageMs > LOST_TIMEOUT_MS)
        {
            state = TrackState::LOST;
        }
        else
        {
            state = TrackState::TRACKING;
        }
    }

    // Convenience helpers
    bool shouldPredict() const
    {
        return state == TrackState::TRACKING || state == TrackState::LOST;
    }

    bool shouldRenderFiltered() const
    {
        return state == TrackState::TRACKING;
    }

    bool isLost() const         { return state == TrackState::LOST; }
    bool hasTarget() const      { return state != TrackState::NO_TARGET; }
    TrackState getState() const { return state; }

private:
    TrackState state;
    std::chrono::steady_clock::time_point lastDetectionTime{};
};
