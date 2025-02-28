#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo/common/Plugin.hh>
#include <ignition/math/Pose3.hh>

namespace gazebo
{
    class CircularMotionPlugin : public ModelPlugin
    {
    private:
        physics::ActorPtr actor;
        event::ConnectionPtr updateConnection;
        double radius;      // 圆周半径(m)
        double angular_vel; // 角速度(rad/s)
        double start_time;  // 起始时间

    public:
        void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf)
        {
            // 获取Actor对象
            this->actor = boost::dynamic_pointer_cast<physics::Actor>(_model);

            // 从SDF读取参数（默认值：半径10m，角速度0.5rad/s）
            this->radius = _sdf->Get<double>("radius", 10.0).first;
            this->angular_vel = _sdf->Get<double>("angular_vel", 0.5).first;

            // 绑定更新事件
            this->updateConnection = event::Events::ConnectWorldUpdateBegin(
                std::bind(&CircularMotionPlugin::OnUpdate, this, std::placeholders::_1));

            gzmsg << "Circular motion plugin loaded: Radius=" << radius
                  << " Angular velocity=" << angular_vel << std::endl;
        }

        void OnUpdate(const common::UpdateInfo &_info)
        {
            // 计算时间差
            double dt = (_info.simTime - start_time).Double();
            start_time = _info.simTime.Double();

            // 计算新位置
            double x = radius * cos(angular_vel * _info.simTime.Double());
            double y = radius * sin(angular_vel * _info.simTime.Double());

            // 计算朝向角度（面向运动方向）
            double yaw = atan2(y, x) + M_PI / 2;

            // 设置Actor位姿
            ignition::math::Pose3d pose(x, y, 0, 0, 0, yaw);
            this->actor->SetWorldPose(pose, true, false);

            // 更新动画（可选）
            this->actor->SetScriptTime(_info.simTime.Double());
        }
    };
    GZ_REGISTER_MODEL_PLUGIN(CircularMotionPlugin)
} // namespace gazebo