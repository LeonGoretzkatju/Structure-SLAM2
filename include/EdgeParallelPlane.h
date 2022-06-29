//
// Created by fishmarch on 19-6-8.
//

#ifndef ORB_SLAM2_EDGEPARALLELPLANE_H
#define ORB_SLAM2_EDGEPARALLELPLANE_H

#include "Thirdparty/g2o/g2o/core/base_vertex.h"
#include "Thirdparty/g2o/g2o/core/hyper_graph_action.h"
#include "Thirdparty/g2o/g2o/core/eigen_types.h"
#include "Thirdparty/g2o/g2o/core/base_binary_edge.h"
#include "Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include "Thirdparty/g2o/g2o/stuff/misc.h"
#include "Plane3D.h"
#include "VertexPlane.h"

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace g2o {
    class EdgeParallelPlane : public BaseBinaryEdge<2, Plane3D, VertexPlane, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeParallelPlane() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[1]);
            const VertexPlane *planeVertex = static_cast<const VertexPlane *>(_vertices[0]);

            const Plane3D &plane = planeVertex->estimate();
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * plane;

            _error = localPlane.ominus_par(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        bool isDepthPositive() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[1]);
            const VertexPlane *planeVertex = static_cast<const VertexPlane *>(_vertices[0]);

            const Plane3D &plane = planeVertex->estimate();
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * plane;

            return localPlane.distance() > 0;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseBinaryEdge::linearizeOplus();
        }
    };

    class EdgeParallelPlaneOnlyPose : public BaseUnaryEdge<2, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeParallelPlaneOnlyPose() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * Xw;

            _error = localPlane.ominus_par(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        bool isDepthPositive() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n * Xw;

            return localPlane.distance() > 0;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseUnaryEdge::linearizeOplus();
        }

        Plane3D Xw;
    };

    class EdgeParallelPlaneOnlyTranslation : public BaseUnaryEdge<2, Plane3D, VertexSE3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeParallelPlaneOnlyTranslation() {}

        void computeError() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);

            // measurement function: remap the plane in global coordinates
            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n + Xc;

            _error = localPlane.ominus_par(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        bool isDepthPositive() {
            const VertexSE3Expmap *poseVertex = static_cast<const VertexSE3Expmap *>(_vertices[0]);

            Isometry3D w2n = poseVertex->estimate();
            Plane3D localPlane = w2n + Xc;

            return localPlane.distance() > 0;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseUnaryEdge::linearizeOplus();

            _jacobianOplusXi(0, 0) = 0;
            _jacobianOplusXi(0, 1) = 0;
            _jacobianOplusXi(0, 2) = 0;
//        _jacobianOplusXi(0, 3) = 0;
//        _jacobianOplusXi(0, 4) = 0;
//        _jacobianOplusXi(0, 5) = 0;

            _jacobianOplusXi(1, 0) = 0;
            _jacobianOplusXi(1, 1) = 0;
            _jacobianOplusXi(1, 2) = 0;
//        _jacobianOplusXi(1, 3) = 0;
//        _jacobianOplusXi(1, 4) = 0;
//        _jacobianOplusXi(1, 5) = 0;
        }

        Plane3D Xc;
    };

    class EdgeParallelPlaneSim3Project : public g2o::BaseBinaryEdge<2, Plane3D, VertexPlane, g2o::VertexSim3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeParallelPlaneSim3Project() {}

        void computeError() {
            const g2o::VertexSim3Expmap *v1 = static_cast<const g2o::VertexSim3Expmap *>(_vertices[1]);
            const g2o::VertexPlane *v2 = static_cast<const g2o::VertexPlane *>(_vertices[0]);

            const Plane3D &plane = v2->estimate();
            g2o::Sim3 sim3 = v1->estimate();

            Vector4D coeffs = plane._coeffs;
            Vector4D localCoeffs;
            Matrix3D R = sim3.rotation().matrix();
            localCoeffs.head<3>() = sim3.scale() * (R * coeffs.head<3>());
            localCoeffs(3) = coeffs(3) - sim3.translation().dot(localCoeffs.head<3>());
            if (localCoeffs(3) < 0.0)
                localCoeffs = -localCoeffs;
            Plane3D localPlane = Plane3D(localCoeffs);

            _error = localPlane.ominus_par(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }
    };

    class EdgeParallelPlaneInverseSim3Project
            : public g2o::BaseBinaryEdge<2, Plane3D, VertexPlane, g2o::VertexSim3Expmap> {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeParallelPlaneInverseSim3Project() {}

        void computeError() {
            const g2o::VertexSim3Expmap *v1 = static_cast<const g2o::VertexSim3Expmap *>(_vertices[1]);
            const g2o::VertexPlane *v2 = static_cast<const g2o::VertexPlane *>(_vertices[0]);

            const Plane3D &plane = v2->estimate();
            g2o::Sim3 sim3 = v1->estimate().inverse();

            Vector4D coeffs = plane._coeffs;
            Vector4D localCoeffs;
            Matrix3D R = sim3.rotation().matrix();
            localCoeffs.head<3>() = sim3.scale() * (R * coeffs.head<3>());
            localCoeffs(3) = coeffs(3) - sim3.translation().dot(localCoeffs.head<3>());
            if (localCoeffs(3) < 0.0)
                localCoeffs = -localCoeffs;
            Plane3D localPlane = Plane3D(localCoeffs);

            _error = localPlane.ominus_par(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }
    };

    class EdgeInertialParallelPlane : public BaseBinaryEdge<2, Plane3D, VertexPlane, ORB_SLAM2::VertexPose> {
        typedef ORB_SLAM2::VertexPose VertexPose;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        EdgeInertialParallelPlane(int cam_idx_=0): cam_idx(cam_idx_){
        }

        void computeError() {
            const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
            const VertexPlane *planeVertex = static_cast<const VertexPlane *>(_vertices[0]);

            const Plane3D &plane = planeVertex->estimate();
            Isometry3D w2n = VPose->estimate().ProjectionMatrix(cam_idx);
            Plane3D localPlane = w2n * plane;

            _error = localPlane.ominus_par(_measurement);
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

        bool isDepthPositive() {
            const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[1]);
            const VertexPlane *planeVertex = static_cast<const VertexPlane *>(_vertices[0]);

            const Plane3D &plane = planeVertex->estimate();
            Isometry3D w2n = VPose->estimate().ProjectionMatrix(cam_idx);
            Plane3D localPlane = w2n * plane;

            return localPlane.distance() > 0;
        }

        virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        virtual void linearizeOplus() {
            BaseBinaryEdge::linearizeOplus();
        }

        public:
            const int cam_idx;
    };

    class EdgeInertialParallelPlaneOnlyPose : public g2o::BaseUnaryEdge<2, Plane3D, ORB_SLAM2::VertexPose>
    {
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;
        typedef ORB_SLAM2::VertexPose VertexPose;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeInertialParallelPlaneOnlyPose(const cv::Mat &Xw_, int cam_idx_=0):Xw(ORB_SLAM2::Converter::toPlane3D(Xw_)),
            cam_idx(cam_idx_){}

         virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        void computeError(){
            const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
            Isometry3D w2n = VPose->estimate().ProjectionMatrix(cam_idx);
            Plane3D localPlane = w2n * Xw;
            _error = localPlane.ominus_par(_measurement);
        }

        void linearizeOplus(){
            BaseUnaryEdge::linearizeOplus();
        }

        bool isDepthPositive() {
            const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
            Isometry3D w2n = VPose->estimate().ProjectionMatrix(cam_idx);
            Plane3D localPlane = w2n * Xw;

            return localPlane.distance() > 0;
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

    public:
        Plane3D Xw;
        // PointCloud::Ptr planePoints;
        const int cam_idx;
    };

    class EdgeInertialParallelPlaneOnlyTranslation : public g2o::BaseUnaryEdge<2, Plane3D, ORB_SLAM2::VertexPose>
    {
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;
        typedef ORB_SLAM2::VertexPose VertexPose;
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        EdgeInertialParallelPlaneOnlyTranslation(const cv::Mat &Xw_, int cam_idx_=0):Xw(ORB_SLAM2::Converter::toPlane3D(Xw_)),
            cam_idx(cam_idx_){}

         virtual bool read(std::istream &is) {
            Vector4D v;
            is >> v(0) >> v(1) >> v(2) >> v(3);
            setMeasurement(Plane3D(v));
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j) {
                    is >> information()(i, j);
                    if (i != j)
                        information()(j, i) = information()(i, j);
                }
            return true;
        }

        virtual bool write(std::ostream &os) const {
            Vector4D v = _measurement.toVector();
            os << v(0) << " " << v(1) << " " << v(2) << " " << v(3) << " ";
            for (int i = 0; i < information().rows(); ++i)
                for (int j = i; j < information().cols(); ++j)
                    os << " " << information()(i, j);
            return os.good();
        }

        void computeError(){
            const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
            Isometry3D w2n = VPose->estimate().ProjectionMatrix(cam_idx);
            Plane3D localPlane = w2n * Xw;
            _error = localPlane.ominus_par(_measurement);
        }

        void linearizeOplus(){
            BaseUnaryEdge::linearizeOplus();

            _jacobianOplusXi(0, 0) = 0;
            _jacobianOplusXi(0, 1) = 0;
            _jacobianOplusXi(0, 2) = 0;

            _jacobianOplusXi(1, 0) = 0;
            _jacobianOplusXi(1, 1) = 0;
            _jacobianOplusXi(1, 2) = 0;
        }

        bool isDepthPositive() {
            const VertexPose* VPose = static_cast<const VertexPose*>(_vertices[0]);
            Isometry3D w2n = VPose->estimate().ProjectionMatrix(cam_idx);
            Plane3D localPlane = w2n * Xw;

            return localPlane.distance() > 0;
        }

        void setMeasurement(const Plane3D &m) {
            _measurement = m;
        }

    public:
        Plane3D Xw;
        // PointCloud::Ptr planePoints;
        const int cam_idx;
    };
}


#endif //ORB_SLAM2_EDGEPARALLELPLANE_H
