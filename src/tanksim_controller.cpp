#include <cmath>
#include <cstdint>
#include <ros/ros.h>
#include <ros/package.h>
#include <raylib.h>
#include <raymath.h>
#include "tanksim/actuator.h"
#include "tanksim/sensor.h"
#include "tanksim/link.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <assert.h>
#include <queue>
#include <algorithm>
//#include <torch/torch.h>
#include <iostream>


/*#define STATE_SIZE 100
#define ACTION_SIZE 4

// ==== Globals ====
torch::Tensor state_tensor;
bool data_ready = false;
sensor_msgs::Image camera_data;
geometry_msgs::Pose2D gps_data;
float health = 1.0;

// === Policy Network (struct-based, not class) ===
struct PolicyNetImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    PolicyNetImpl(int input_size, int hidden_size, int output_size){
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x){
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        return fc3->forward(x);
    }
};
TORCH_MODULE(PolicyNet);

torch::nn::Sequential net = torch::nn::Sequential(
    torch::nn::Linear(STATE_SIZE, 128),
    torch::nn::ReLU(),
    torch::nn::Linear(128, 128),
    torch::nn::ReLU(),
    torch::nn::Linear(128, ACTION_SIZE),
    torch::nn::Tanh()
);

// === Build a simple flat state tensor ===
torch::Tensor build_state() {
    std::vector<float> state(STATE_SIZE, 0.0f);

    // Fill some example data:
    state[0] = gps_data.x;
    state[1] = gps_data.y;
    state[2] = gps_data.theta;
    state[3] = health;

    // TODO: process image data and encode enemy locations etc

    return torch::from_blob(state.data(), {1, STATE_SIZE}).clone();
}

// === Publish action ===
void publish_action(const torch::Tensor& action_tensor) {
    auto* a = action_tensor.data_ptr<float>();

    std_msgs::Float32MultiArray drive;
    drive.data.push_back(a[0]); // left throttle
    drive.data.push_back(a[1]); // right throttle
    drive_pub.publish(drive);

    std_msgs::Float32MultiArray turret;
    turret.data.push_back(a[2]); // turret turn
    turret.data.push_back(a[3] > 0.8f ? 1.0f : 0.0f); // fire
    turret_pub.publish(turret);
}*/

/// STATE AND SENSORS

enum e_ros_sensors {
	ROS_SENSOR_GPS,
    ROS_SENSOR_IMU,
    ROS_SENSOR_DIAG,
    ROS_SENSOR_LIDAR,
    ROS_SENSOR_CAMERA,
    ROS_SENSOR_TURRET,

	ROS_SENSOR_LEN
};

const char* ros_sensors[] = {
	"sensor/gps",
	"sensor/imu",
	"sensor/diag",
	"sensor/lidar",
	"sensor/camera",
	"sensor/turret"
};

enum e_ros_actuators {
	ROS_ACTUATOR_DRIVE,
	ROS_ACTUATOR_TURRET,

	ROS_ACTUATOR_LEN
};

const char* ros_actuators[] = {
	"actuator/drive",
	"actuator/twist"
};

// invisible heuristic state input, storing 'most threatening target' or whatever
typedef struct {

} istate_t;

// visible heuristic state input tensor for GOAP and RL
typedef struct {

} vstate_t;

// heuristic output tensor
typedef struct {

} weight_t;

typedef struct {
	vstate_t precondition;
	vstate_t postcondition;
	float scalar; // rl backed cost scalar
} action_t;

typedef struct {
	// individual tank agents
	// friendly info (ie health) updated by central planner
	// enemy info updated by group vision, hit tracking etc.
	size_t id;
	float health,turret_ammo;
	float damage_given;
	float damage_taken;

	Vector3 position;
	Vector3 vel_linear;
	Vector3 vel_angular;
	Quaternion orientation;
	Quaternion turret_orientation;

	uint8_t los, turret_ready;
} agent_t;

typedef struct {
	int w,h,scalar;
	float* heightfield;
} map_t;

// world state
#define LIDAR_BIN 8
typedef struct {
	map_t map; // updated by local observation and central planner

	agent_t self;
	float lidar[LIDAR_BIN];
	std::vector<agent_t> friendlies; // updated real time
	std::vector<agent_t> combatants; // updated by local observation and central planner
} slam_t;

typedef struct {
    double kp,ki,kd;
    double perr,integral,set;
} controller_pid_t;

enum pid_types {
	CONTROLLER_PID_MAIN_LINEAR,
	CONTROLLER_PID_MAIN_ANGULAR,
	CONTROLLER_PID_AUX_ANGULAR_PITCH,
	CONTROLLER_PID_AUX_ANGULAR_YAW,

	CONTROLLER_PID_LEN
};

typedef struct {
	controller_pid_t pid[CONTROLLER_PID_LEN];
} controller_t;

typedef struct {
	map_t* map;
	size_t* parents;
	float* costs;
	std::vector<std::pair<float,size_t>> open;
	std::vector<Vector2> path;
	size_t cell_current, cell_target, cell_target_last, cell_last_valid, path_idx;
} mp_planner_t;

// constant controller state nested giga struct
typedef struct {
	ros::Subscriber subscribers[ROS_SENSOR_LEN];
    ros::Publisher publishers[ROS_ACTUATOR_LEN];
    std::vector<float> sensors[ROS_SENSOR_LEN];

    float sim_speed;
    slam_t* slam; // local knowledge
    mp_planner_t* planner; // motion planning
    controller_t* controller;
} ros_t;

inline float util_ema(float a, float b, float alpha){
	return alpha*b+(1.0f-alpha)*(a);
}

#define TAU 2*M_PI
void ros_sensor_callback(size_t type, std::vector<float> data, ros_t* ros){
	slam_t* slam = ros->slam;
	agent_t* self = &slam->self;
	float dt = ros->sim_speed;

	// switch for sensors
	// lidar updates terrain map
	switch(type){
		case ROS_SENSOR_LIDAR:{
			//printf("lidar!\n");
			// 90 samples returning distance and relative quaternion angle ( 5 floats )
			// assuming everything valid of course
			// proof of concept lidar SLAM in milsim
			size_t n = 0, o = 0;
			for(size_t i = 0; i < data.size(); i += 5){
				float d = data[i];
				Quaternion q_sensor = (Quaternion){data[i+1],data[i+2],data[i+3],data[i+4]};
				Quaternion q_abs = QuaternionMultiply(self->orientation,q_sensor);
				Vector3 v_direction = (Vector3){0,0,1};
				v_direction = Vector3RotateByQuaternion(v_direction,q_abs);
				Vector3 v_pos = Vector3Add(self->position,Vector3Scale(v_direction,d));

				// 1. Project to XZ plane
			    Vector3 dir_xz = {v_direction.x, 0, v_direction.z};
			    float len = Vector3Length(dir_xz);
			    if (len < 0.001f) continue; // Avoid degenerate straight-down rays

			    // 2. Compute azimuth
			    float angle = atan2f(dir_xz.z, dir_xz.x); // atan2(z,x) gives CW angle from +X
			    if (angle < 0) angle += TAU; // Normalize to [0, 2PI)

			    // 3. Bin index
			    int bin = (int)(angle / (TAU/LIDAR_BIN))%LIDAR_BIN;
			    slam->lidar[bin] = util_ema(d,slam->lidar[bin],.5);

				int xi = (int)v_pos.x/slam->map.scalar + slam->map.w/2;
				int zi = (int)v_pos.z/slam->map.scalar + slam->map.h/2;
				if (xi < 0 || xi >= slam->map.w || zi < 0 || zi >= slam->map.h) continue;
				float* h = &slam->map.heightfield[(zi) * slam->map.w + (xi)];
				if(*h < 0){
					*h = v_pos.y; // first sample
					n++;
				}else{
					*h = std::max(*h,v_pos.y);//util_ema(*h,v_pos.y,.5); // exponential moving average
					if(*h == v_pos.y){
						// if lidar is higher, inflate surrounding terrain
						const int radius = 2;
						for (int dx = -radius; dx <= radius; dx++) {
						    for (int dz = -radius; dz <= radius; dz++) {
						        int xi_n = xi + dx;
						        int zi_n = zi + dz;
						        if (xi_n < 0 || xi_n >= slam->map.w || zi_n < 0 || zi_n >= slam->map.h) continue;

						        float* h2 = &slam->map.heightfield[zi_n * slam->map.w + xi_n];
						        if(*h2 < 0){
						            *h2 = v_pos.y;
						        }else{
						            float dist = sqrtf(dx * dx + dz * dz);
						            float decay = 1.0f / (1.0f + dist);
						            *h2 = util_ema(*h2, v_pos.y * decay,.5);
						        }
						    }
						}
					}
					o++;
				}
			}
			//printf("lidar: %zu reads, %zu new cells, %zu updated cells\n",data.size()/5,n,o);
		} break;
		case ROS_SENSOR_IMU:{
			// orientation
			self->orientation = (Quaternion){data[0],data[1],data[2],data[3]};

			// angular velocity
			self->vel_angular = (Vector3){data[4],data[5],data[6]};

			// linear velocity and fractional position value by acceleration
			// /Vector3 v_velocity_local = (Vector3){data[7]/dt,data[8]/dt,data[9]/dt};
			if (dt <= 0.0001f || std::isnan(dt) || std::isinf(dt)) {
		        ROS_ERROR("Invalid dt = %f in IMU sensor callback. Skipping integration.", dt);
		        break;
		    }

		    Vector3 v_velocity_local = (Vector3){data[7]/dt,data[8]/dt,data[9]/dt};

		    if (!std::isfinite(v_velocity_local.x) || !std::isfinite(v_velocity_local.y) || !std::isfinite(v_velocity_local.z)) {
		        ROS_ERROR("IMU linear velocity is non-finite: (%f, %f, %f)", v_velocity_local.x, v_velocity_local.y, v_velocity_local.z);
		        break;
		    }
			self->vel_linear = Vector3RotateByQuaternion(v_velocity_local, self->orientation);
			self->position = Vector3Add(self->position,self->vel_linear);

			//printf("imu read: q: %f %f %f %f, av: %f %f %f, lv: %f %f %f\n",
			//	data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7],data[8],data[9]);
		} break;
		case ROS_SENSOR_GPS:{
			//Vector3 v_frac = (Vector3){fmodf(self->position.x,1),fmodf(self->position.y,1),fmodf(self->position.z,1)};
			Vector3 v_gps = (Vector3){data[0],data[1],data[2]};
			float error = Vector3Distance(self->position,v_gps);
			const float max_error = 2.;
			float alpha = std::clamp(error/max_error,0.1f,0.9f);
			if(error > max_error) self->position = v_gps; // hard reset
			else self->position = (Vector3){util_ema(self->position.x,v_gps.x,alpha),util_ema(self->position.y,v_gps.y,alpha),util_ema(self->position.z,v_gps.z,alpha)};
			//printf("gps: %f %f %f, up: %f %f %f\n",data[0],data[1],data[2],self->position.x,self->position.y,self->position.z);
		} break;
		case ROS_SENSOR_CAMERA:{
			// track enemies TODO
		} break;
		case ROS_SENSOR_DIAG:{
			// track health
			self->health = data[0];
		} break;
		case ROS_SENSOR_TURRET:{
			// track ammo, turret orientation and reload timing
			self->turret_orientation = (Quaternion){data[0],data[1],data[2],data[3]};
			self->turret_ammo = data[4];
			self->turret_ready = (uint8_t)data[5];
		} break;
	}
}


/// MOTION PLANNING

inline size_t mp_get_idx(int x, int z, map_t* map){
	if(x < 0 || x >= map->w || z < 0 || z >= map->h) return SIZE_MAX;
	return z*map->w+x;
}

inline Vector2 mp_get_vec(size_t idx, map_t* map){
	return (Vector2){(float)(idx%map->w),(float)(idx/map->h)};
}

inline size_t mp_remap_idx(Vector2 vec, map_t* map){ // world space coords to grid cell
	Vector2 v_shift = Vector2Scale((Vector2){(float)map->w/2,(float)map->h/2},map->scalar);
	Vector2 v_scaled = Vector2Scale(Vector2Add(vec,v_shift),1.0f/map->scalar);
	return mp_get_idx((int)v_scaled.x,(int)v_scaled.y,map);
}

inline Vector2 mp_remap_vec(size_t idx, map_t* map){ // grid cell to world space coords
	Vector2 vec = mp_get_vec(idx,map);
	Vector2 v_shift = Vector2Scale((Vector2){(float)map->w/2,(float)map->h/2},map->scalar);
	return Vector2Subtract(Vector2Scale(vec,map->scalar),v_shift);
}

void mp_path(mp_planner_t* planner, size_t start, size_t end) {
	planner->path.clear();
    size_t current = end;
    Vector2 v_shift = (Vector2){(float)planner->map->w,(float)planner->map->h};
    size_t count = 1;
    printf("PATH:[");
    while(current != start){
    	// rescale vectors
    	Vector2 vec = mp_remap_vec(current,planner->map);
    	printf("(%.0f,%.0f),",vec.x,vec.y);
        planner->path.push_back(vec);//Vector2Subtract(Vector2Scale(mp_remap_vec(current,planner->map),(float)planner->map->w),v_shift));
        current = planner->parents[current];
        count++;
    }
    printf("]\n");
    // add start
    //planner->path.push_back(mp_remap_vec(start,planner->map));
    planner->path_idx = planner->path.size()-1;
    printf("new path: len %zu\n",count);
}

inline void mp_heap_push(std::vector<std::pair<float,size_t>>& heap, std::pair<float,size_t> val) {
    heap.push_back(val);
    std::push_heap(heap.begin(), heap.end(), std::greater<>());
}

inline std::pair<float,size_t> mp_heap_pop(std::vector<std::pair<float,size_t>>& heap) {
    std::pop_heap(heap.begin(), heap.end(), std::greater<>());
    auto val = heap.back();
    heap.pop_back();
    return val;
}

inline float mp_astar_euclidean(size_t a, size_t b, map_t* map){
	Vector2 v_a = mp_get_vec(a,map);
	Vector2 v_b = mp_get_vec(b,map);
	return Vector2Distance(v_a,v_b);
}

void mp_astar(mp_planner_t* planner, size_t sidx, size_t eidx){
	if(planner->parents == NULL) planner->parents = (size_t*)calloc(planner->map->w*planner->map->h,sizeof(size_t));
	if(planner->costs == NULL) planner->costs = (float*)calloc(planner->map->w*planner->map->h,sizeof(float));
	planner->open.clear();
	// /memset(planner->parents,0,planner->map->w*planner->map->h*sizeof(size_t));
	//memset(planner->costs,0,planner->map->w*planner->map->h*sizeof(float));
	for(size_t i = 0; i < planner->map->w*planner->map->h; i++){
		planner->parents[i] = SIZE_MAX;
		planner->costs[i] = INFINITY;
	}

	/*Vector2 v_shift = (Vector2){(float)planner->map->w/2,(float)planner->map->h/2};
	Vector2 v_start_scaled = Vector2Scale(Vector2Add(start,v_shift),1.0f/planner->map->scalar);
	Vector2 v_end_scaled = Vector2Scale(Vector2Add(end,v_shift),1.0f/planner->map->scalar);
	size_t sidx = util_get_idx((int)v_start_scaled.x,(int)v_start_scaled.y,planner->map);
	size_t eidx = util_get_idx((int)v_end_scaled.x,(int)v_end_scaled.y,planner->map);*/

	mp_heap_push(planner->open,{mp_astar_euclidean(sidx,eidx,planner->map),sidx});
	planner->parents[sidx] = sidx;
	planner->costs[sidx] = 0;
	size_t cell_min = sidx;
	float dist_min = -1;
	uint8_t path_found = 0;
	size_t count = 0;

	while(!planner->open.empty()){
		count++;

        auto [fscore, current] = mp_heap_pop(planner->open);
        if(current == eidx){
            mp_path(planner,sidx,eidx);
            path_found = 1;
            break;
        }

        // ignore if path to this node is already shorter
        //if(planner->costs[current]+mp_astar_euclidean(current,eidx,planner->map) <= fscore) continue;
        //planner->costs[current] = fscore;
        Vector2 v_current = mp_get_vec(current,planner->map);
        float y = planner->map->heightfield[current];
        float g_current = planner->costs[current];

        int cx = current%planner->map->w;
        int cy = current/planner->map->w;

        // 8 direction astar with euclidean distance
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = cx + dx;
                int ny = cy + dy;
                size_t idx = mp_get_idx(nx,ny,planner->map);
                if (idx == SIZE_MAX) continue; // idx not in heightfield
                float e = planner->map->heightfield[idx];
                float g_scalar = e-y;
                if(e == 0) g_scalar = 1;
                if(g_scalar > 1.f){
                	//planner->costs[current] += g_scalar/4.;
                	continue; // grade too steep
                }

                float g_score = ((dx == 0 || dy == 0)?1.f:1.414f)*(1+fabsf(g_scalar));
                float tentative_g = g_current + g_score; // cost from current to neighbor
                if(planner->costs[idx] <= tentative_g) continue;
                planner->costs[idx] = tentative_g;
                float h = mp_astar_euclidean(idx,eidx,planner->map);
                float f = tentative_g+h;
                if(dist_min == -1 || h < dist_min){
                	cell_min = idx;
                	dist_min = h;
                }

                mp_heap_push(planner->open, {f, idx});
                planner->parents[idx] = current;
            }
        }
    }
    if(path_found == 0) mp_path(planner,sidx,cell_min);
    printf("evaluated %zu cells\n",count);
}

float util_wrap_angle(float angle) {
    while(angle > M_PI) angle -= TAU;
    while(angle < -M_PI) angle += TAU;
    return angle;
}

float controller_pid(controller_pid_t* pid, float error, float dt){
    pid->integral = std::clamp(pid->integral+error*dt,-1.,1.);
    double derivative = (error-pid->perr)/dt;
    double output = std::clamp(pid->kp*error + pid->ki*pid->integral + pid->kd*derivative,-1.,1.);
    pid->perr = error;
    return output;
}

void mp_travel(ros_t* ros){
	controller_t* controller = ros->controller;
	mp_planner_t* planner = ros->planner;
	agent_t* self = &ros->slam->self;
	slam_t* slam = ros->slam;
	Vector2 v_pos = (Vector2){self->position.x,self->position.z};
	size_t sidx = mp_remap_idx(v_pos,planner->map);

	size_t target = planner->cell_target;
	/*if(planner->map->heightfield[sidx]-self->position.y > 2.){
		// tank is in degenerate cell
		printf("tank in degenerate cell! t:%.1f c:%.1f\n",self->position.y,planner->map->heightfield[sidx]);
		target = planner->cell_last_valid;
	}else planner->cell_last_valid = sidx;*/
	if(sidx != planner->cell_current || planner->cell_target != planner->cell_target_last) mp_astar(planner,sidx,target);
	planner->cell_target_last = planner->cell_target;
	planner->cell_current = sidx;

	if(planner->cell_current == planner->cell_target){
		printf("end travel!!!\n");
		planner->cell_target = rand()%(planner->map->w*planner->map->h);
	}	

	Vector2 v_pos_goal = mp_remap_vec(planner->cell_target,planner->map);
	Vector2 v_pos_path = v_pos;
	float v_distance_goal = Vector2Distance(v_pos_goal,v_pos);
	if(planner->path_idx > 0){
		//if(planner->path.size() <= planner->path_idx) break;
		v_pos_path = planner->path[planner->path_idx];
		if(Vector2Distance(v_pos,v_pos_path) < planner->map->scalar*2) planner->path_idx--;
		//else if(Vector2Distance(v_pos_goal,v_pos_path) > v_distance_goal) planner->path_idx--;
	}

	float v_distance = Vector2Distance(v_pos,v_pos_path);
	Vector3 e_dir = QuaternionToEuler(self->orientation);
	Vector2 v_dir_goal = Vector2Normalize(Vector2Subtract(v_pos_path, v_pos));
	//Vector2 v_dir_forward = { cosf(e_dir.y), sinf(e_dir.y) };
	Vector3 forward_world = Vector3RotateByQuaternion((Vector3){0, 0, 1}, self->orientation);
	Vector2 v_dir_forward = Vector2Normalize((Vector2){ forward_world.x, forward_world.z });
	float dot = v_dir_goal.x * v_dir_forward.x + v_dir_goal.y * v_dir_forward.y;
	float det = v_dir_goal.x * v_dir_forward.y - v_dir_goal.y * v_dir_forward.x;
	float angle_difference = util_wrap_angle(atan2f(det, dot));
	if(fabsf(dot + 1.0f) < 0.01f){
	    // goal is almost exactly opposite direction
	    angle_difference = M_PI;
	}
	/*float angle_difference = atan2f(v_dir_goal.y, v_dir_goal.x) - atan2f(v_dir_forward.y, v_dir_forward.x);
	while (angle_difference > M_PI) angle_difference -= TAU;
	while (angle_difference < -M_PI) angle_difference += TAU;*/

	printf("goal dir: %.2f %.2f | forward: %.2f %.2f | angle diff: %.1f deg\n",
       v_dir_goal.x, v_dir_goal.y,
       v_dir_forward.x, v_dir_forward.y,
       angle_difference * 180.0f / M_PI);

	//printf("path idx: %zu, path len: %zu, dist to goal: %f, dist to next path idx: %f\n",planner->path_idx,planner->path.size(),v_distance_goal,v_distance);
	Vector2 cell_pos = mp_remap_vec(sidx,planner->map);
	//printf("cell: %f %f pos %f %f\n",cell_pos.x,cell_pos.y,v_pos.x,v_pos.y);

	// angular pid
	float steering = controller_pid(&controller->pid[CONTROLLER_PID_MAIN_ANGULAR],angle_difference,ros->sim_speed);
	float throttle = controller_pid(&controller->pid[CONTROLLER_PID_MAIN_LINEAR],v_distance,ros->sim_speed)*std::fmax(cos(angle_difference),0);

	//printf("angle diff: %.2f deg | throttle: %.2f | steering: %.2f\n", angle_difference * RAD2DEG, throttle, steering);

	int bin_index = (int)((e_dir.y + M_PI)/(TAU/LIDAR_BIN))%LIDAR_BIN;
	float obstacle_distance = slam->lidar[bin_index];
	float min_safe_distance = 8.0f; // meters, tweak as needed
	float brake = 0;
	/*printf("obstacle dist: ");
	for(size_t i = 0; i < 32; i++) printf("%.0f, ",slam->lidar[i]);
	printf("\n");*/
	if (obstacle_distance > 0 && obstacle_distance < min_safe_distance) {
	    throttle = 0;
	    if(planner->path_idx < planner->path.size()-1) planner->path_idx++;
	    printf("degenerate obstacle, braking and stepping back in path\n");
	}

	float left = std::clamp(throttle + steering, -1.f, 1.f);
	float right = std::clamp(throttle - steering, -1.f, 1.f);

	tanksim::actuator msg;
	msg.actuator.push_back(left);
	msg.actuator.push_back(right);

	msg.actuator.push_back(brake);
	msg.actuator.push_back(brake);
	ros->publishers[ROS_ACTUATOR_DRIVE].publish(msg);

	//printf("pos: %f %f %f\n",self->position.x,self->position.y,self->position.z);

	// random position travel
	//if(rand()%600 == 0) planner->cell_target = rand()%(planner->map->w*planner->map->h);
}


int main(int argc, char** argv){
	std::string node_name = "tank_controller_" + std::to_string(getpid());
    ros::init(argc, argv,node_name);
    ros::NodeHandle nh_anon;
    srand(time(NULL));

    ros::ServiceClient client = nh_anon.serviceClient<tanksim::link>("link");
	tanksim::link srv;
	std::string ns;
	if (!client.call(srv)) {
		printf("SIMULATOR NOT RUNNING\n");
		return 1;
	}
    uint32_t id = srv.response.id;
    float sim_speed = srv.response.speed;
    uint8_t team = srv.response.team;
    uint32_t map_width = srv.response.width;
    uint32_t map_height = srv.response.height;
    ns = "tank_" + std::to_string(id);
    printf("linked to namespace %s\n",ns.c_str());
    ros::NodeHandle nh(ns);
    // env init
    ros_t ros; 
    ros.sim_speed = sim_speed;
    slam_t slam = {0};
    slam.map.scalar = 2;
    slam.map.w = map_width/slam.map.scalar;
    slam.map.h = map_height/slam.map.scalar;
    slam.map.heightfield = (float*)calloc(slam.map.w*slam.map.h,sizeof(float));

    mp_planner_t mp_planner = {0};
    mp_planner.map = &slam.map;
    mp_planner.cell_target = rand()%(mp_planner.map->w*mp_planner.map->h);

    controller_t controller = {0};
    controller.pid[CONTROLLER_PID_MAIN_ANGULAR] = (controller_pid_t){.5,.01,.1, 0.,0.,2.};
    controller.pid[CONTROLLER_PID_MAIN_LINEAR] = (controller_pid_t){1.,.05,.2, 0.,0.,5.};


    ros.slam = &slam;
    ros.planner = &mp_planner;
    ros.controller = &controller;
    ros_t* rosptr = &ros;

    for(size_t i = 0; i < ROS_SENSOR_LEN; i++){
    	ros.subscribers[i] = nh.subscribe<tanksim::sensor>(
	        ros_sensors[i], 1,
	        [i, rosptr](const tanksim::sensor::ConstPtr& msg) { ros_sensor_callback(i, msg->data, rosptr); } // more evil pointer capture
	    );
	    printf("subscribed to %s\n",ros_sensors[i]);
    }

	for(size_t i = 0; i < ROS_ACTUATOR_LEN; i++){
		ros.publishers[i] = nh.advertise<tanksim::actuator>(ros_actuators[i],1);
		printf("advertising %s\n",ros_actuators[i]);
	}

    ros::Rate loop((double)sim_speed); // 60hz in real time, faster for training

    while (ros::ok()) {
        ros::spinOnce();
        mp_travel(rosptr);

        /*if (data_ready) {
            state_tensor = build_state();
            torch::Tensor action = net->forward(state_tensor);
            publish_action(action);
            data_ready = false;
        }*/

        loop.sleep();
    }

    return 0;
}
