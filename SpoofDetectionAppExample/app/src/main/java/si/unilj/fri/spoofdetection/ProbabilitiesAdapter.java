package si.unilj.fri.spoofdetection;

import android.animation.ObjectAnimator;
import android.annotation.SuppressLint;
import android.content.res.ColorStateList;
import android.view.LayoutInflater;
import android.view.ViewGroup;
import android.view.animation.AccelerateDecelerateInterpolator;

import androidx.annotation.NonNull;
import androidx.core.util.Pair;
import androidx.recyclerview.widget.RecyclerView;

import org.tensorflow.lite.support.label.Category;

import java.util.ArrayList;

import si.unilj.fri.spoofdetection.databinding.ItemProbabilityBinding;

public class ProbabilitiesAdapter extends RecyclerView.Adapter<ProbabilitiesAdapter.ViewHolder> {
    ArrayList<Category> categoryList = new ArrayList<>();

    static class ViewHolder extends RecyclerView.ViewHolder {
        private final ItemProbabilityBinding binding;
        private final ArrayList<Pair<ColorStateList, ColorStateList>> progressColorPairList = new ArrayList<>();

        public ViewHolder(ItemProbabilityBinding binding) {
            super(binding.getRoot());
            this.binding = binding;
            this.progressColorPairList.add(Pair.create(ColorStateList.valueOf(0xfff9e7e4), ColorStateList.valueOf(0xffd97c2e)));
            this.progressColorPairList.add(Pair.create(ColorStateList.valueOf(0xfff7e3e8), ColorStateList.valueOf(0xffc95670)));
            this.progressColorPairList.add(Pair.create(ColorStateList.valueOf(0xffecf0f9), ColorStateList.valueOf(0xff714Fe7)));
        }

        @SuppressLint("DefaultLocale")
        void bind(int position, String label, Float score) {
            binding.labelTextView.setText(label);
            binding.scoreText.setText(String.format("%.4f", score));
            binding.progressBar.setProgressBackgroundTintList(this.progressColorPairList.get(position % 3).first);
            binding.progressBar.setProgressTintList(this.progressColorPairList.get(position % 3).second);
            int newValue = (int)(score * 100);
//            binding.progressBar.setProgress(newValue);

            // If you don't want to animate, you can write like `progressBar.progress = newValue`.;
            ObjectAnimator animation = ObjectAnimator.ofInt(binding.progressBar, "progress", binding.progressBar.getProgress(), newValue);
            animation.setDuration(100);
            animation.setInterpolator(new AccelerateDecelerateInterpolator());
            animation.start();
        }
    }

    public ProbabilitiesAdapter() {
    }

    @Override
    @NonNull
    public ViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        return new ViewHolder(ItemProbabilityBinding.inflate(LayoutInflater.from(parent.getContext())));
    }

    @Override
    public void onBindViewHolder(@NonNull ViewHolder holder, int position) {
        Category category = categoryList.get(position);
        holder.bind(position, category.getLabel(), category.getScore());
    }

    @Override
    public int getItemCount() {
        return categoryList.size();
    }
}
